/// Real sparse CSC matrix for writing to a MAT file (borrowed slices)
pub struct SparseCSC<'a, T> {
    /// Row index of each non-zero element
    pub row_indices: &'a [u32],
    /// Column pointer array (length = ncols + 1)
    pub col_pointers: &'a [u32],
    /// Non-zero values
    pub values: &'a [T],
    /// Matrix dimensions [nrows, ncols]
    pub dims: [usize; 2],
}

impl<'a, T> SparseCSC<'a, T> {
    pub fn new(
        row_indices: &'a [u32],
        col_pointers: &'a [u32],
        values: &'a [T],
        dims: [usize; 2],
    ) -> Self {
        assert_eq!(
            col_pointers.len(),
            dims[1] + 1,
            "col_pointers length must be ncols + 1: {} != {}",
            col_pointers.len(),
            dims[1] + 1
        );
        assert_eq!(
            values.len(),
            row_indices.len(),
            "values and row_indices must have the same length: {} != {}",
            values.len(),
            row_indices.len()
        );
        Self {
            row_indices,
            col_pointers,
            values,
            dims,
        }
    }
}

/// Real sparse CSC matrix read from a MAT file (owned Vecs)
pub struct SparseCSCOwned<T> {
    pub row_indices: Vec<u32>,
    pub col_pointers: Vec<u32>,
    pub values: Vec<T>,
    pub dims: [usize; 2],
}

/// Complex sparse CSC matrix for writing to a MAT file (borrowed slices)
pub struct ComplexSparseCSC<'a, T> {
    /// Row index of each non-zero element
    pub row_indices: &'a [u32],
    /// Column pointer array (length = ncols + 1)
    pub col_pointers: &'a [u32],
    /// Real parts of non-zero values
    pub re: &'a [T],
    /// Imaginary parts of non-zero values
    pub im: &'a [T],
    /// Matrix dimensions [nrows, ncols]
    pub dims: [usize; 2],
}

impl<'a, T> ComplexSparseCSC<'a, T> {
    pub fn new(
        row_indices: &'a [u32],
        col_pointers: &'a [u32],
        re: &'a [T],
        im: &'a [T],
        dims: [usize; 2],
    ) -> Self {
        assert_eq!(
            col_pointers.len(),
            dims[1] + 1,
            "col_pointers length must be ncols + 1: {} != {}",
            col_pointers.len(),
            dims[1] + 1
        );
        assert_eq!(
            re.len(),
            im.len(),
            "real and imaginary parts must have the same length: {} != {}",
            re.len(),
            im.len()
        );
        assert_eq!(
            re.len(),
            row_indices.len(),
            "values and row_indices must have the same length: {} != {}",
            re.len(),
            row_indices.len()
        );
        Self {
            row_indices,
            col_pointers,
            re,
            im,
            dims,
        }
    }
}

/// Complex sparse CSC matrix read from a MAT file (owned Vecs)
pub struct ComplexSparseCSCOwned<T> {
    pub row_indices: Vec<u32>,
    pub col_pointers: Vec<u32>,
    pub re: Vec<T>,
    pub im: Vec<T>,
    pub dims: [usize; 2],
}
