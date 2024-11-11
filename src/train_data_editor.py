import os
import pandas as pd
import json
import streamlit as st


# Function to load and display all files in a directory ending with '_train_data'
def load_train_data_files(path):
    train_dirs = []
    train_files = []
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            if dir.endswith("_train_data"):
                full_path = os.path.join(root, dir)
                train_dirs.append(full_path)

    for train_dir in train_dirs:
        for root, dirs, files in os.walk(train_dir):
            for file in files:
                if file.endswith(".csv"):
                    full_path = os.path.join(root, file)
                    train_files.append(full_path)

    return train_files


# Function to load a file and parse it into a DataFrame
def load_file(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


# Function to save edited data back to CSV
def save_file(data, file_path):
    try:
        data.to_csv(file_path, index=False)
        st.success(f"File saved successfully to {file_path}")
    except Exception as e:
        st.error(f"Error saving file: {e}")


# Streamlit App
def main():
    st.title("Train Data Editor")

    # Folder selection
    path = st.text_input("Enter the directory path:", value=f"{os.getcwd()}/knowledge")

    if os.path.isdir(path):
        files = load_train_data_files(path)

        if files:
            st.write(f"Found {len(files)} files.")

            # Select file to edit
            selected_file = st.selectbox("Select a file to edit:", files)

            if selected_file:
                data = load_file(selected_file)

                if data is not None:
                    st.write(f"Editing file: {selected_file}")

                    # Ensure the file has the necessary columns
                    if 'conversations' in data.columns and 'score' in data.columns:
                        # Editable table
                        edited_data = []

                        # Add row functionality
                        if st.button("Add Row"):
                            new_row = {
                                'conversations': json.dumps([
                                    {"from": "human", "value": ""},
                                    {"from": "gpt", "value": ""}
                                ]),
                                'source': '',
                                'score': 0.0,
                                'split': ''
                            }
                            # Add the new row to the dataframe
                            data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)

                        # Create editable table
                        for idx, row in data.iterrows():
                            conv = json.loads(row['conversations'])

                            # Extract the human and gpt conversation parts
                            human_value = conv[0]['value']
                            gpt_value = conv[1]['value']
                            score_value = row['score']

                            cols = st.columns([2, 2, 1, 1])  # Add an extra column for the "Action" button

                            with cols[0]:
                                human_input = st.text_area(f"Human (Row {idx + 1})", value=human_value, key=f"human_{idx}")
                            with cols[1]:
                                gpt_input = st.text_area(f"GPT (Row {idx + 1})", value=gpt_value, key=f"gpt_{idx}")
                            with cols[2]:
                                score_input = st.number_input(f"Score (Row {idx + 1})", value=score_value, step=0.1, key=f"score_{idx}")

                            # Update the conversation and score with the new inputs
                            conv[0]['value'] = human_input
                            conv[1]['value'] = gpt_input
                            row['conversations'] = json.dumps(conv)
                            row['score'] = score_input

                            # Action column for removing the row
                            with cols[3]:
                                remove_button = st.button(f"Remove", key=f"remove_{idx}")
                                if remove_button:
                                    data = data.drop(idx)
                                    data.reset_index(drop=True, inplace=True)
                                    # Rerun the app to refresh the table
                                    st.rerun(scope='app')

                            # Collect the updated row into a list
                            edited_data.append(row)

                        # Convert the edited data back to a DataFrame using concat
                        edited_df = pd.concat([pd.DataFrame([row]) for row in edited_data], ignore_index=True)

                        # Save edited data
                        if st.button("Save Changes"):
                            save_file(edited_df, selected_file)
                    else:
                        st.error("Selected file does not have a 'conversations' or 'score' column.")
        else:
            st.error(f"No files ending with '_train_data' found in {path}.")
    else:
        st.error(f"Invalid directory: {path}")


if __name__ == "__main__":
    main()
