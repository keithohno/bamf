pub enum Node<T> {
    Children(Vec<Node<T>>),
    Leaf(T),
}

impl<T: Copy> From<Vec<Node<T>>> for Node<T> {
    fn from(vec: Vec<Node<T>>) -> Node<T> {
        Node::Children(vec)
    }
}

impl<T: Copy> From<Vec<T>> for Node<T> {
    fn from(vec: Vec<T>) -> Node<T> {
        Node::Children(vec.iter().map(|x| Node::Leaf(*x)).collect())
    }
}

pub fn flatten<T: Copy>(vec: &Node<T>) -> Vec<T> {
    match vec {
        Node::Children(children) => children.iter().map(|x| flatten(x)).flatten().collect(),
        Node::Leaf(leaf) => vec![*leaf],
    }
}
