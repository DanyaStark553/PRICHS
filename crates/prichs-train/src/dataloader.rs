use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;
use std::ptr::NonNull;
use std::slice;

pub const MAGIC: [u8; 8] = *b"PRICHS01";
pub const HEADER_SIZE: usize = 64;

#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrichsBinHeader {
    pub magic: [u8; 8],
    pub total_tokens: u64,
    pub vocab_size: u32,
    pub eos_token_id: u16,
    pub _reserved: [u8; 42],
}

#[derive(Debug)]
pub enum DataError {
    Io(std::io::Error),
    InvalidMagic,
    FileTooSmall,
    MmapFailed,
    OutOfBounds,
}

impl std::fmt::Display for DataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataError::Io(e) => write!(f, "IO error: {}", e),
            DataError::InvalidMagic => write!(f, "Invalid magic number in header"),
            DataError::FileTooSmall => write!(f, "File too small to be valid"),
            DataError::MmapFailed => write!(f, "Mmap mapping failed"),
            DataError::OutOfBounds => write!(f, "Sequence slice out of bounds"),
        }
    }
}

impl std::error::Error for DataError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DataError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for DataError {
    fn from(e: std::io::Error) -> Self {
        DataError::Io(e)
    }
}

/// Zero-Copy Document-Aware Shard Writer
pub struct PrichsShardWriter {
    writer: BufWriter<File>,
    total_tokens: u64,
    vocab_size: u32,
    eos_token_id: u16,
}

impl PrichsShardWriter {
    pub fn create<P: AsRef<Path>>(path: P, vocab_size: u32, eos_token_id: u16) -> Result<Self, DataError> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;

        let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, file);
        let placeholder = [0u8; HEADER_SIZE];
        writer.write_all(&placeholder)?;

        Ok(Self {
            writer,
            total_tokens: 0,
            vocab_size,
            eos_token_id,
        })
    }

    #[inline]
    pub fn append_document(&mut self, tokens: &[u16]) -> Result<(), DataError> {
        let byte_slice = unsafe {
            slice::from_raw_parts(tokens.as_ptr() as *const u8, tokens.len() * 2)
        };
        self.writer.write_all(byte_slice)?;
        self.writer.write_all(&self.eos_token_id.to_le_bytes())?;
        self.total_tokens += (tokens.len() + 1) as u64;
        Ok(())
    }

    pub fn finalize(mut self) -> Result<u64, DataError> {
        self.writer.flush()?;
        let mut file = self.writer.into_inner().map_err(|e| DataError::Io(e.into_error()))?;

        let header = PrichsBinHeader {
            magic: MAGIC,
            total_tokens: self.total_tokens,
            vocab_size: self.vocab_size,
            eos_token_id: self.eos_token_id,
            _reserved: [0u8; 42],
        };

        let header_bytes = unsafe {
            slice::from_raw_parts(&header as *const PrichsBinHeader as *const u8, HEADER_SIZE)
        };

        file.seek(SeekFrom::Start(0))?;
        file.write_all(header_bytes)?;
        file.sync_all()?;

        Ok(self.total_tokens)
    }
}

/// Zero-Copy Document-Aware Mmap Dataset
pub struct PrichsMmapDataset {
    #[cfg(unix)]
    ptr: NonNull<u8>,
    #[cfg(unix)]
    file_size: usize,

    #[cfg(not(unix))]
    _mmap: memmap2::Mmap,
    #[cfg(not(unix))]
    ptr: NonNull<u8>,

    pub header: PrichsBinHeader,
}

unsafe impl Send for PrichsMmapDataset {}
unsafe impl Sync for PrichsMmapDataset {}

impl PrichsMmapDataset {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, DataError> {
        let file = File::open(path)?;
        let file_size = file.metadata()?.len() as usize;

        if file_size < HEADER_SIZE {
            return Err(DataError::FileTooSmall);
        }

        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            let raw_ptr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    file_size,
                    libc::PROT_READ,
                    libc::MAP_PRIVATE | libc::MAP_POPULATE,
                    file.as_raw_fd(),
                    0,
                )
            };

            if raw_ptr == libc::MAP_FAILED {
                return Err(DataError::MmapFailed);
            }

            unsafe {
                libc::madvise(raw_ptr, file_size, libc::MADV_SEQUENTIAL | libc::MADV_WILLNEED);
            }

            let header = unsafe { *(raw_ptr as *const PrichsBinHeader) };
            if header.magic != MAGIC {
                unsafe { libc::munmap(raw_ptr, file_size) };
                return Err(DataError::InvalidMagic);
            }

            let ptr = NonNull::new(raw_ptr as *mut u8).ok_or(DataError::MmapFailed)?;
            Ok(Self { ptr, file_size, header })
        }

        #[cfg(not(unix))]
        {
            let mmap = unsafe { memmap2::Mmap::map(&file)? };
            let raw_ptr = mmap.as_ptr() as *mut u8;

            let header = unsafe { *(raw_ptr as *const PrichsBinHeader) };
            if header.magic != MAGIC {
                return Err(DataError::InvalidMagic);
            }

            let ptr = NonNull::new(raw_ptr).ok_or(DataError::MmapFailed)?;
            Ok(Self { _mmap: mmap, ptr, header })
        }
    }

    #[inline]
    pub fn tokens(&self) -> &[u16] {
        unsafe {
            let data_ptr = self.ptr.as_ptr().add(HEADER_SIZE) as *const u16;
            slice::from_raw_parts(data_ptr, self.header.total_tokens as usize)
        }
    }

    /// Returns (inputs, targets, cu_seqlens) for FlashAttention-2 varlen masking
    /// completely eliminating cross-document attention noise without <pad> tokens.
    pub fn get_sequence_with_boundaries(
        &self,
        start_idx: usize,
        seq_len: usize,
    ) -> Result<(&[u16], &[u16], Vec<i32>), DataError> {
        let all_tokens = self.tokens();
        if start_idx + seq_len + 1 > all_tokens.len() {
            return Err(DataError::OutOfBounds);
        }

        let x = &all_tokens[start_idx..start_idx + seq_len];
        let y = &all_tokens[start_idx + 1..start_idx + seq_len + 1];

        let mut cu_seqlens = Vec::with_capacity(8);
        cu_seqlens.push(0);
        for (i, &tok) in x.iter().enumerate() {
            if tok == self.header.eos_token_id && i + 1 < seq_len {
                cu_seqlens.push((i + 1) as i32);
            }
        }
        cu_seqlens.push(seq_len as i32);

        Ok((x, y, cu_seqlens))
    }
}

pub const SFT_MAGIC: [u8; 8] = *b"PRICHSFT";
pub const SFT_HEADER_SIZE: usize = 64;

#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrichsSftHeader {
    pub magic: [u8; 8],
    pub num_samples: u32,
    pub seq_len: u32,
    pub vocab_size: u32,
    pub eos_token_id: u16,
    pub _reserved: [u8; 42],
}

/// Zero-Copy Document-Isolated SFT Dataset with Prompt & Padding Masking
pub struct PrichsSftDataset {
    #[cfg(unix)]
    ptr: NonNull<u8>,
    #[cfg(unix)]
    file_size: usize,

    #[cfg(not(unix))]
    _mmap: memmap2::Mmap,
    #[cfg(not(unix))]
    ptr: NonNull<u8>,

    pub header: PrichsSftHeader,
}

unsafe impl Send for PrichsSftDataset {}
unsafe impl Sync for PrichsSftDataset {}

impl PrichsSftDataset {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, DataError> {
        let file = File::open(path)?;
        let file_size = file.metadata()?.len() as usize;

        if file_size < SFT_HEADER_SIZE {
            return Err(DataError::FileTooSmall);
        }

        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            let raw_ptr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    file_size,
                    libc::PROT_READ,
                    libc::MAP_PRIVATE | libc::MAP_POPULATE,
                    file.as_raw_fd(),
                    0,
                )
            };

            if raw_ptr == libc::MAP_FAILED {
                return Err(DataError::MmapFailed);
            }

            let header = unsafe { *(raw_ptr as *const PrichsSftHeader) };
            if header.magic != SFT_MAGIC {
                unsafe { libc::munmap(raw_ptr, file_size) };
                return Err(DataError::InvalidMagic);
            }

            let ptr = NonNull::new(raw_ptr as *mut u8).ok_or(DataError::MmapFailed)?;
            Ok(Self { ptr, file_size, header })
        }

        #[cfg(not(unix))]
        {
            let mmap = unsafe { memmap2::Mmap::map(&file)? };
            let raw_ptr = mmap.as_ptr() as *mut u8;

            let header = unsafe { *(raw_ptr as *const PrichsSftHeader) };
            if header.magic != SFT_MAGIC {
                return Err(DataError::InvalidMagic);
            }

            let ptr = NonNull::new(raw_ptr).ok_or(DataError::MmapFailed)?;
            Ok(Self { _mmap: mmap, ptr, header })
        }
    }

    #[inline]
    pub fn num_samples(&self) -> usize {
        self.header.num_samples as usize
    }

    #[inline]
    pub fn seq_len(&self) -> usize {
        self.header.seq_len as usize
    }

    /// Sample record size in bytes: seq_len * 2 (inputs) + seq_len * 2 (targets) + seq_len * 1 (mask)
    #[inline]
    pub fn sample_bytes(&self) -> usize {
        let s = self.seq_len();
        s * 2 + s * 2 + s
    }

    /// Returns zero-copy (&inputs, &targets, &mask_u8) for dialogue at index `sample_idx`
    pub fn get_sample(&self, sample_idx: usize) -> Result<(&[u16], &[u16], &[u8]), DataError> {
        if sample_idx >= self.num_samples() {
            return Err(DataError::OutOfBounds);
        }

        let s = self.seq_len();
        let sample_size = self.sample_bytes();
        let offset = SFT_HEADER_SIZE + sample_idx * sample_size;

        unsafe {
            let base_ptr = self.ptr.as_ptr().add(offset);
            let inputs_ptr = base_ptr as *const u16;
            let targets_ptr = base_ptr.add(s * 2) as *const u16;
            let mask_ptr = base_ptr.add(s * 4) as *const u8;

            let inputs = slice::from_raw_parts(inputs_ptr, s);
            let targets = slice::from_raw_parts(targets_ptr, s);
            let mask = slice::from_raw_parts(mask_ptr, s);

            Ok((inputs, targets, mask))
        }
    }
}

pub struct PrichsSftWriter {
    writer: BufWriter<File>,
    num_samples: u32,
    seq_len: u32,
    vocab_size: u32,
    eos_token_id: u16,
}

impl PrichsSftWriter {
    pub fn create<P: AsRef<Path>>(
        path: P,
        seq_len: u32,
        vocab_size: u32,
        eos_token_id: u16,
    ) -> Result<Self, DataError> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;

        let mut writer = BufWriter::with_capacity(4 * 1024 * 1024, file);
        let placeholder = [0u8; SFT_HEADER_SIZE];
        writer.write_all(&placeholder)?;

        Ok(Self {
            writer,
            num_samples: 0,
            seq_len,
            vocab_size,
            eos_token_id,
        })
    }

    pub fn append_dialogue(
        &mut self,
        inputs: &[u16],
        targets: &[u16],
        mask: &[u8],
    ) -> Result<(), DataError> {
        let s = self.seq_len as usize;
        assert_eq!(inputs.len(), s, "inputs length must match seq_len");
        assert_eq!(targets.len(), s, "targets length must match seq_len");
        assert_eq!(mask.len(), s, "mask length must match seq_len");

        let inputs_bytes = unsafe {
            slice::from_raw_parts(inputs.as_ptr() as *const u8, s * 2)
        };
        let targets_bytes = unsafe {
            slice::from_raw_parts(targets.as_ptr() as *const u8, s * 2)
        };

        self.writer.write_all(inputs_bytes)?;
        self.writer.write_all(targets_bytes)?;
        self.writer.write_all(mask)?;
        self.num_samples += 1;
        Ok(())
    }

    pub fn finalize(mut self) -> Result<u32, DataError> {
        self.writer.flush()?;
        let mut file = self.writer.into_inner().map_err(|e| DataError::Io(e.into_error()))?;

        let header = PrichsSftHeader {
            magic: SFT_MAGIC,
            num_samples: self.num_samples,
            seq_len: self.seq_len,
            vocab_size: self.vocab_size,
            eos_token_id: self.eos_token_id,
            _reserved: [0u8; 42],
        };

        let header_bytes = unsafe {
            slice::from_raw_parts(&header as *const PrichsSftHeader as *const u8, SFT_HEADER_SIZE)
        };

        file.seek(SeekFrom::Start(0))?;
        file.write_all(header_bytes)?;
        file.sync_all()?;

        Ok(self.num_samples)
    }
}

#[cfg(unix)]
impl Drop for PrichsMmapDataset {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr.as_ptr() as *mut libc::c_void, self.file_size);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pretrain_v2_dataset_integrity() {
        let candidates = [
            "../../data/pretrain_v2_large.prichsbin",
            "data/pretrain_v2_large.prichsbin",
            "../../data/pretrain_v2.prichsbin",
            "data/pretrain_v2.prichsbin",
        ];
        let mut tested = 0;
        for &p_str in &candidates {
            let path = Path::new(p_str);
            if !path.exists() {
                continue;
            }
            let dataset = PrichsMmapDataset::open(path).expect("Failed to open prichsbin");
            assert_eq!(&dataset.header.magic, b"PRICHS01");
            assert_eq!(dataset.header.vocab_size, 32768);
            assert_eq!(dataset.header.eos_token_id, 2);
            assert!(dataset.header.total_tokens >= 6_000_000);

            let seq_len = 1024;
            let (inputs, targets, has_boundary) = dataset.get_sequence_with_boundaries(0, seq_len)
                .expect("Failed to get first sequence");
            assert_eq!(inputs.len(), seq_len);
            assert_eq!(targets.len(), seq_len);

            for &tok in inputs {
                assert!((tok as u32) < dataset.header.vocab_size, "Token {} exceeds vocab size", tok);
            }
            for &tok in targets {
                assert!((tok as u32) < dataset.header.vocab_size, "Target {} exceeds vocab size", tok);
            }

            println!("Dataset {:?} check OK: total_tokens = {}, boundaries = {:?}", 
                p_str, dataset.header.total_tokens, has_boundary);
            tested += 1;
        }
        assert!(tested > 0, "No pretrain dataset found to test");
    }
}
