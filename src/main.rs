use polars::prelude::*;

fn main() {
    let arr1 = &[1, 2, 3];
    let arr2 = &[4, 5, 6];

    let df = DataFrame::from_iter(vec![Series::new("x", arr1), Series::new("y", arr2)]);

    for s in df["arr1"].iter() {
        dbg!(s);
    }
}
