# Plan for Implementing an Evaluation Loop

This plan outlines the steps to implement an evaluation loop for the "Denoising Dirty Documents" Kaggle competition.

1. **Create `PLAN.md`:** Create a new file `PLAN.md` to document the plan.
2. **Implement an evaluation loop in `evaluate.py`:**
    - Load the test data.
    - For each test image, generate the denoised version using the trained model.
    - Convert the denoised image into the required CSV format for submission. The format will be `id,value`, where `id` is `image_id_row_col` and `value` is the predicted pixel intensity.
    - Append the results to a `submission.csv` file.
3. **Update `data_loader.py` to handle test data:**
    - Add a function to load the test images.
4. **Update `helpers.py`:**
    - Add a function to convert an image to the CSV submission format.
5. **Run the evaluation script:**
    - Generate the `submission.csv` file.
