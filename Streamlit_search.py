import streamlit as st
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import os

# ✅ Handle NumPy 2.0 Compatibility Issue
import numpy as np
if not hasattr(np, 'float_'):
    np.float_ = np.float64  # Ensure compatibility with chromadb

# ✅ Database and client setup
db_path = r"C:\anuman\Khooj\image_vdb"  # Your DB path

# Initialize chromadb client
client = chromadb.PersistentClient(path=db_path)
embedding_function = OpenCLIPEmbeddingFunction()
data_loader = ImageLoader()

# ✅ Create or get collection
collection = client.get_or_create_collection(
    name='multimodal_collection2',
    embedding_function=embedding_function,
    data_loader=data_loader
)

# ✅ Display banner image
banner_image_path = r"C:\anuman\Khooj\img.jpg"  # Path to banner image
if os.path.exists(banner_image_path):
    st.image(banner_image_path, use_column_width=True)
else:
    st.warning("Banner image not found!")

# ✅ Streamlit UI
st.title("🚗 Vehicle Image Search Engine")

# Search bar
query = st.text_input("Enter your search query:")
parent_path = r"C:\anuman\Khooj\vehicle_images\train\images"  # Image folder path

if st.button("Search"):
    if query.strip():  # Check if query is not empty
        try:
            # ✅ Querying the collection
            results = collection.query(query_texts=[query], n_results=5, include=["distances"])

            if results and 'ids' in results and 'distances' in results:
                for image_id, distance in zip(results['ids'][0], results['distances'][0]):
                    image_path = os.path.join(parent_path, image_id)
                    
                    if os.path.exists(image_path):
                        st.image(image_path, caption=os.path.basename(image_path))
                        st.write(f"Distance: {distance:.4f}")
                    else:
                        st.warning(f"Image not found: {image_id}")
            else:
                st.write("No matching results found.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a search query.")
