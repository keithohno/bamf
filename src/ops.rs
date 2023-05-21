pub trait Scale<S> {
    fn scale(&self, arg: S) -> Self;
}
