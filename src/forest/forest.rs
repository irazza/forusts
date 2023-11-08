pub trait Forest {
    fn fit(&mut self, x: &Vec<Vec<f64>>, y: &Vec<usize>);
    fn predict(&self, x: &Vec<Vec<f64>>) -> Vec<usize>;
    fn pairwise_breiman(&self, x1: Vec<Vec<f64>>, x2: Vec<Vec<f64>>) -> Vec<Vec<f64>>;
    fn pairwise_ancestor(&self, x1: Vec<Vec<f64>>, x2: Vec<Vec<f64>>) -> Vec<Vec<f64>>;
    fn pairwise_zhu(&self, x1: Vec<Vec<f64>>, x2: Vec<Vec<f64>>) -> Vec<Vec<f64>>;
}
