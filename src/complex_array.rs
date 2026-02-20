/// Complex dense array for writing to a MAT file (borrowed slices, zero-copy to FFI)
pub struct ComplexArray<'a, T> {
    pub re: &'a [T],
    pub im: &'a [T],
    pub dims: Vec<u64>,
}

impl<'a, T> ComplexArray<'a, T> {
    /// Creates a new complex dense array
    ///
    /// `re` and `im` must have the same length, equal to the product of `dims`.
    pub fn new(re: &'a [T], im: &'a [T], dims: Vec<u64>) -> Self {
        let n: u64 = dims.iter().product();
        assert_eq!(
            re.len(),
            im.len(),
            "real and imaginary parts must have the same length: {} != {}",
            re.len(),
            im.len()
        );
        assert_eq!(
            n,
            re.len() as u64,
            "expected {} elements, found {}",
            n,
            re.len() as u64
        );
        Self { re, im, dims }
    }
}

/// Complex dense array read from a MAT file (owned Vecs)
pub struct ComplexVec<T> {
    pub re: Vec<T>,
    pub im: Vec<T>,
    pub dims: Vec<usize>,
}
