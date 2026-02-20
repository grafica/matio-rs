use crate::{
    complex_array::ComplexVec,
    sparse::{ComplexSparseCSCOwned, SparseCSCOwned},
    DataType, Mat, MatioError, Result,
};
use std::ptr;

/// Convert a [Mat] variable into a Rust data type
pub trait MayBeInto<T> {
    fn maybe_into(self) -> Result<T>;
}
macro_rules! maybe_into {
    ( $( $rs:ty ),+ ) => {
	    $(

            impl<'a> MayBeInto<$rs> for &Mat<'a> {
                fn maybe_into(self) -> Result<$rs> {
                    if self.len() > 1 {
                        return Err(MatioError::Scalar(self.name.clone(), self.len()));
                    }
                    match self.mat_type() {
                        Some(mat) if <$rs as DataType>::mat_type() == mat => {
                            Ok(unsafe { ((*self.matvar_t).data as *mut $rs).read() })
                        }
                        _ => Err(MatioError::TypeMismatch(
                            self.name.clone(),
                            <$rs as DataType>::to_string(),
                            self.mat_type().map(|t| t.to_string()).unwrap_or_default(),
                        )),
                    }
                }
            }

        impl<'a> MayBeInto<$rs> for Mat<'a> {
            fn maybe_into(self) -> Result<$rs> {
                <&Mat<'a> as MayBeInto<$rs>>::maybe_into(&self)
            }
        }

        impl<'a> MayBeInto<Vec<$rs>> for &Mat<'a> {
            fn maybe_into(self) -> Result<Vec<$rs>> {
                match self.mat_type() {
                    Some(mat) if <$rs as DataType>::mat_type() == mat => {
                        let n = self.len();
                        let mut value: Vec<$rs> = Vec::with_capacity(n);
                        unsafe {
                            ptr::copy((*self.matvar_t).data as *mut $rs, value.as_mut_ptr(), n);
                            value.set_len(n);
                        }
                        Ok(value)
                    }
                    _ => Err(MatioError::TypeMismatch(
                        self.name.clone(),
                        <$rs as DataType>::to_string(),
                        self.mat_type().map(|t| t.to_string()).unwrap_or_default(),
                    )),
                }
            }
        }


        impl<'a> MayBeInto<Vec<$rs>> for Mat<'a> {
            fn maybe_into(self) -> Result<Vec<$rs>> {
                <&Mat<'a> as MayBeInto<Vec<$rs>>>::maybe_into(&self)
            }
        }

        #[cfg(feature = "nalgebra")]
        impl<'a>
            MayBeInto<nalgebra::DMatrix<$rs>> for &Mat<'a>
        {
            fn maybe_into(self) -> Result<nalgebra::DMatrix<$rs>> {
                match self.mat_type() {
                    Some(mat) if <$rs as DataType>::mat_type() == mat => {
                        if self.rank() > 2 {
                            return Err(MatioError::Rank(self.rank()));
                        }
                        let dims = self.dims();
                        let (nrows, ncols) = (dims[0] as usize, dims[1] as usize);
                        let data: Vec<$rs> = self.maybe_into()?;
                        Ok(nalgebra::DMatrix::from_column_slice(
                            nrows,
                            ncols,
                            data.as_slice(),
                        ))
                    }
                    _ => Err(MatioError::TypeMismatch(
                        self.name.clone(),
                        <$rs as DataType>::to_string(),
                        self.mat_type().map(|t| t.to_string()).unwrap_or_default(),
                    )),
                }
            }
        }

        #[cfg(feature = "nalgebra")]
        impl<'a>
        MayBeInto<nalgebra::DMatrix<$rs>> for Mat<'a>
        {
            fn maybe_into(self) -> Result<nalgebra::DMatrix<$rs>> {
                <&Mat<'a> as MayBeInto<nalgebra::DMatrix<$rs>>>::maybe_into(&self)
            }
        }

        #[cfg(feature = "faer")]
        impl<'a>
            MayBeInto<faer::mat::Mat<$rs>> for &Mat<'a>
        {
            fn maybe_into(self) -> Result<faer::mat::Mat<$rs>> {
                match self.mat_type() {
                    Some(mat) if <$rs as DataType>::mat_type() == mat => {
                        if self.rank() > 2 {
                            return Err(MatioError::Rank(self.rank()));
                        }
                        let dims = self.dims();
                        let (nrows, ncols) = (dims[0] as usize, dims[1] as usize);
                        let data: Vec<$rs> = self.maybe_into()?;
                        let mat = faer::MatRef::from_column_major_slice(data.as_slice(), nrows, ncols);
                        Ok(mat.cloned())
                    }
                    _ => Err(MatioError::TypeMismatch(
                        self.name.clone(),
                        <$rs as DataType>::to_string(),
                        self.mat_type().map(|t| t.to_string()).unwrap_or_default(),
                    )),
                }
            }
        }

        #[cfg(feature = "faer")]
        impl<'a>
            MayBeInto<faer::mat::Mat<$rs>> for Mat<'a>
        {
            fn maybe_into(self) -> Result<faer::mat::Mat<$rs>> {
                <&Mat<'a> as MayBeInto<faer::mat::Mat<$rs>>>::maybe_into(&self)
            }
        }

        )+
    };
}

maybe_into! {
    f64,
    f32,
    i8,
    i16,
    i32,
    i64,
    u8,
    u16,
    u32,
    u64
}

impl<'a> MayBeInto<Mat<'a>> for Mat<'a> {
    fn maybe_into(self) -> Result<Mat<'a>> {
        Ok(self)
    }
}

impl<'a> MayBeInto<String> for Mat<'a> {
    fn maybe_into(self) -> Result<String> {
        match self.mat_type() {
            Some(mat) if <String as DataType>::mat_type() == mat => {
                let n = self.len();
                let mut value: Vec<u8> = Vec::with_capacity(n);
                unsafe {
                    ptr::copy((*self.matvar_t).data as *mut u8, value.as_mut_ptr(), n);
                    value.set_len(n);
                }
                Ok(String::from_utf8(value)?)
            }
            _ => Err(MatioError::TypeMismatch(
                self.name.clone(),
                <String as DataType>::to_string(),
                self.mat_type().map(|t| t.to_string()).unwrap_or_default(),
            )),
        }
    }
}

impl<'a> MayBeInto<Vec<String>> for Mat<'a> {
    fn maybe_into(self) -> Result<Vec<String>> {
        match self.mat_type() {
            Some(mat) if <Vec<String> as DataType>::mat_type() == mat => {
                let n = self.len();
                let mut value: Vec<String> = Vec::with_capacity(n);
                for i in 0..n {
                    let matvar_t = unsafe { ffi::Mat_VarGetCell(self.matvar_t, i as i32) };
                    let mat = Mat::as_ptr(String::new(), matvar_t)?;
                    let rs = <Mat<'a> as MayBeInto<String>>::maybe_into(mat)?;
                    value.push(rs);
                }
                Ok(value)
            }
            _ => Err(MatioError::TypeMismatch(
                self.name.clone(),
                <Vec<String> as DataType>::to_string(),
                self.mat_type().map(|t| t.to_string()).unwrap_or_default(),
            )),
        }
    }
}

// --- Complex dense MayBeInto ---

macro_rules! maybe_into_complex {
    ( $( ($rs:ty,$mat_c:expr,$mat_t:expr) ),+ ) => {
        $(
            impl<'a> MayBeInto<ComplexVec<$rs>> for Mat<'a> {
                fn maybe_into(self) -> Result<ComplexVec<$rs>> {
                    if !self.is_complex() {
                        return Err(MatioError::TypeMismatch(
                            self.name.clone(),
                            "complex".to_string(),
                            "real".to_string(),
                        ));
                    }
                    let data_type = unsafe { (*self.matvar_t).data_type };
                    if data_type != $mat_t {
                        return Err(MatioError::TypeMismatch(
                            self.name.clone(),
                            stringify!($rs).to_string(),
                            self.mat_type().map(|t| t.to_string()).unwrap_or_default(),
                        ));
                    }
                    let n = self.len();
                    let dims = self.dims();
                    let complex_split = unsafe {
                        &*((*self.matvar_t).data as *const ffi::mat_complex_split_t)
                    };
                    let mut re: Vec<$rs> = Vec::with_capacity(n);
                    let mut im: Vec<$rs> = Vec::with_capacity(n);
                    unsafe {
                        ptr::copy(complex_split.Re as *const $rs, re.as_mut_ptr(), n);
                        ptr::copy(complex_split.Im as *const $rs, im.as_mut_ptr(), n);
                        re.set_len(n);
                        im.set_len(n);
                    }
                    Ok(ComplexVec { re, im, dims })
                }
            }
        )+
    };
}

maybe_into_complex! {
    (f64, ffi::matio_classes_MAT_C_DOUBLE, ffi::matio_types_MAT_T_DOUBLE),
    (f32, ffi::matio_classes_MAT_C_SINGLE, ffi::matio_types_MAT_T_SINGLE)
}

// --- Sparse MayBeInto ---

macro_rules! maybe_into_sparse {
    ( $( ($rs:ty,$mat_t:expr) ),+ ) => {
        $(
            impl<'a> MayBeInto<SparseCSCOwned<$rs>> for Mat<'a> {
                fn maybe_into(self) -> Result<SparseCSCOwned<$rs>> {
                    if !self.is_sparse() {
                        return Err(MatioError::TypeMismatch(
                            self.name.clone(),
                            "sparse".to_string(),
                            "dense".to_string(),
                        ));
                    }
                    let data_type = unsafe { (*self.matvar_t).data_type };
                    if data_type != $mat_t {
                        return Err(MatioError::TypeMismatch(
                            self.name.clone(),
                            stringify!($rs).to_string(),
                            self.mat_type().map(|t| t.to_string()).unwrap_or_default(),
                        ));
                    }
                    let dims = self.dims();
                    let sparse = unsafe {
                        &*((*self.matvar_t).data as *const ffi::mat_sparse_t)
                    };
                    let nir = sparse.nir as usize;
                    let njc = sparse.njc as usize;
                    let ndata = sparse.ndata as usize;

                    let mut row_indices: Vec<u32> = Vec::with_capacity(nir);
                    let mut col_pointers: Vec<u32> = Vec::with_capacity(njc);
                    let mut values: Vec<$rs> = Vec::with_capacity(ndata);
                    unsafe {
                        ptr::copy(sparse.ir, row_indices.as_mut_ptr(), nir);
                        row_indices.set_len(nir);
                        ptr::copy(sparse.jc, col_pointers.as_mut_ptr(), njc);
                        col_pointers.set_len(njc);
                        ptr::copy(sparse.data as *const $rs, values.as_mut_ptr(), ndata);
                        values.set_len(ndata);
                    }
                    Ok(SparseCSCOwned {
                        row_indices,
                        col_pointers,
                        values,
                        dims: [dims[0], dims[1]],
                    })
                }
            }

            impl<'a> MayBeInto<ComplexSparseCSCOwned<$rs>> for Mat<'a> {
                fn maybe_into(self) -> Result<ComplexSparseCSCOwned<$rs>> {
                    if !self.is_sparse() {
                        return Err(MatioError::TypeMismatch(
                            self.name.clone(),
                            "complex sparse".to_string(),
                            "dense".to_string(),
                        ));
                    }
                    if !self.is_complex() {
                        return Err(MatioError::TypeMismatch(
                            self.name.clone(),
                            "complex sparse".to_string(),
                            "real sparse".to_string(),
                        ));
                    }
                    let data_type = unsafe { (*self.matvar_t).data_type };
                    if data_type != $mat_t {
                        return Err(MatioError::TypeMismatch(
                            self.name.clone(),
                            stringify!($rs).to_string(),
                            self.mat_type().map(|t| t.to_string()).unwrap_or_default(),
                        ));
                    }
                    let dims = self.dims();
                    let sparse = unsafe {
                        &*((*self.matvar_t).data as *const ffi::mat_sparse_t)
                    };
                    let nir = sparse.nir as usize;
                    let njc = sparse.njc as usize;
                    let ndata = sparse.ndata as usize;
                    let complex_split = unsafe {
                        &*(sparse.data as *const ffi::mat_complex_split_t)
                    };

                    let mut row_indices: Vec<u32> = Vec::with_capacity(nir);
                    let mut col_pointers: Vec<u32> = Vec::with_capacity(njc);
                    let mut re: Vec<$rs> = Vec::with_capacity(ndata);
                    let mut im: Vec<$rs> = Vec::with_capacity(ndata);
                    unsafe {
                        ptr::copy(sparse.ir, row_indices.as_mut_ptr(), nir);
                        row_indices.set_len(nir);
                        ptr::copy(sparse.jc, col_pointers.as_mut_ptr(), njc);
                        col_pointers.set_len(njc);
                        ptr::copy(complex_split.Re as *const $rs, re.as_mut_ptr(), ndata);
                        re.set_len(ndata);
                        ptr::copy(complex_split.Im as *const $rs, im.as_mut_ptr(), ndata);
                        im.set_len(ndata);
                    }
                    Ok(ComplexSparseCSCOwned {
                        row_indices,
                        col_pointers,
                        re,
                        im,
                        dims: [dims[0], dims[1]],
                    })
                }
            }
        )+
    };
}

maybe_into_sparse! {
    (f64, ffi::matio_types_MAT_T_DOUBLE),
    (f32, ffi::matio_types_MAT_T_SINGLE)
}
