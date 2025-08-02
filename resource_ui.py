import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import gradio as gr
class ResourceMatcher:
    def __init__(self):
        print("Initializing BERT model...")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.eval()
        self.resources = []
        print("Model initialized!")
    def get_bert_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.numpy()
    def add_resource(self, type, name, description, owner, status, contact):
        resource_info = {
            'type': type,
            'name': name,
            'description': description,
            'owner': owner,
            'status': status,
            'contact': contact
        }
        description_text = f"{type} {name} {description}"
        embedding = self.get_bert_embedding(description_text)
        resource_info['embedding'] = embedding
        self.resources.append(resource_info)
        return "Resource added successfully!"
    def search_resources(self, query):
        if not self.resources:
            return "No resources available in the system."
        query_embedding = self.get_bert_embedding(query)
        matches = []
        for resource in self.resources:
            similarity = cosine_similarity(query_embedding, resource['embedding'])[0][0]
            if similarity > 0.5:  # Threshold for matching
                matches.append({
                    'resource': resource,
                    'similarity': similarity
                })
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        if not matches:
            return "No matching resources found."
        results = []
        for idx, match in enumerate(matches, 1):
            resource = match['resource']
            result = f"""
Match #{idx} (Similarity: {match['similarity']:.2f})
Type: {resource['type']}
Name: {resource['name']}
Description: {resource['description']}
Owner: {resource['owner']}
Status: {resource['status']}
Contact: {resource['contact']}
-------------------"""
            results.append(result)
        return "\n".join(results)
matcher = ResourceMatcher()
def add_resource_interface(type, name, description, owner, status, contact):
    try:
        return matcher.add_resource(type, name, description, owner, status, contact)
    except Exception as e:
        return f"Error: {str(e)}"

def search_resources_interface(query):
    return matcher.search_resources(query)
with gr.Blocks(title="Resource Sharing Platform") as demo:
    gr.Markdown("# Resource Sharing Platform")
    with gr.Tab("Add Resource"):
        gr.Markdown("## Add a New Resource")
        with gr.Row():
            with gr.Column():
                type_input = gr.Dropdown(
                    choices=["book", "notes", "hardware", "other"],
                    label="Resource Type"
                )
                name_input = gr.Textbox(label="Resource Name")
                description_input = gr.Textbox(label="Description", lines=3)
                owner_input = gr.Textbox(label="Your Name")
                status_input = gr.Radio(
                    choices=["lending", "giveaway"],
                    label="Status"
                )
                contact_input = gr.Textbox(label="Contact Information")
                add_btn = gr.Button("Add Resource")
                result_output = gr.Textbox(label="Result")
        add_btn.click(
            add_resource_interface,
            inputs=[type_input, name_input, description_input,
                   owner_input, status_input, contact_input],
            outputs=result_output
        )
    with gr.Tab("Search Resources"):
        gr.Markdown("## Search for Resources")
        with gr.Row():
            with gr.Column():
                query_input = gr.Textbox(
                    label="What are you looking for?",
                    placeholder="Example: DSP textbook for signals course"
                )
                search_btn = gr.Button("Search")
                search_output = gr.Textbox(label="Results", lines=10)

        search_btn.click(
            search_resources_interface,
            inputs=query_input,
            outputs=search_output
        )

# Launch the interface
if __name__ == "__main__":
    demo.launch()
