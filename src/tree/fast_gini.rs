use hashbrown::HashMap;

#[derive(Clone)]
pub struct FastGini {
    classes: HashMap<isize, usize>,
    p_sum: u128,
    total: usize,
}

impl FastGini {
    pub fn new() -> Self {
        Self {
            classes: HashMap::new(),
            p_sum: 0,
            total: 0,
        }
    }
    pub fn from_classes_count(classes: HashMap<isize, usize>) -> Self {
        let mut p_sum = 0;
        let mut total = 0;
        for mult in classes.values() {
            total += *mult;
            let mult = *mult as u128;
            p_sum += mult * mult;
        }

        Self {
            classes,
            p_sum,
            total,
        }
    }

    pub fn change_element(&mut self, class: isize, amount: isize) {
        let counter = self.classes.entry(class).or_insert(0);

        let last_counter = *counter as u128;
        self.p_sum -= last_counter * last_counter;

        *counter = counter.saturating_add_signed(amount);

        let new_counter = *counter as u128;

        self.p_sum += new_counter * new_counter;

        self.total = self.total.saturating_add_signed(amount);
    }

    pub fn get_gini(&mut self) -> f64 {
        1.0 - (self.p_sum as f64 / (self.total * self.total) as f64)
    }
}
