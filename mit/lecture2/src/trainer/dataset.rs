// Let the loader select patient indices without copying the stored dataset.
use crate::common::data::EncodedData;

impl<F: Send + Sync, I: Send + Sync> burn::data::dataset::Dataset<usize> for EncodedData<F, I> {
    fn get(&self, index: usize) -> Option<usize> {
        if index < self.targets.len() {
            Some(index)
        } else {
            None
        }
    }

    fn len(&self) -> usize {
        self.targets.len()
    }
}
