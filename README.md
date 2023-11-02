# Easy weed mask

This project is a tool developed to facilitate the creation of image masks that highlight the presence of common chickweed based on various vegetation indices. Common chickweed (Stellaria media) is a common weed in fields and gardens, and these masks can be useful for identifying and monitoring its presence in satellite images or field photographs.

## Supported Vegetation Indices

The tool supports various vegetation indices that can be used to detect the presence of common chickweed. The included vegetation indices are:

1. **Color Index of Vegetation (CIVE)**: A vegetation color index that highlights green areas.

2. **Excessive Green (ExG)**: An index that measures the amount of green in the images.

3. **Excessive Red (ExR)**: An index that measures the amount of red in the images.

4. **Normalized Difference Index (NDI)**: A normalized difference index that compares near-infrared and red bands.

5. **Excessive Red Minus Excessive Green (ExGR)**: Calculates the difference between excessive red and excessive green.

6. **Vegetative Index (VEG)**: An index that highlights vegetative areas in the images.

7. **Combined Indices (COM)**: A combination of multiple indices to improve detection.

## How to Use

Here are the basic steps for using this tool:

1. **Clone the Project**: Start by cloning the project to your local machine.

2. **Run the Application**: Launch the application by running the following command in your terminal:

   ```bash
   streamlit run app.py

This command will start the application using Streamlit.

3. **Select the Dataset Path**: In the application interface, you will have the option to select the path to your dataset. All the JPG images within the chosen directory will be listed.

4. **Generate Masks**: Once you've selected the dataset path, the tool will automatically process and list all the JPG images within the directory. You can then proceed to generate masks for the images using the available vegetation indices.

5. **Save the Masks**: The generated masks can be saved for later use or analysis

## Requirements

Make sure you have the following requirements installed before running the tool:

- [Python](https://www.python.org/)
- Required Python libraries (listed in the `requirements.txt` file)

## Contributions

Contributions are welcome! If you wish to contribute to this project, feel free to open an issue, submit a pull request, or provide feedback.
