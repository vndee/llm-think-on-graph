import streamlit as st
import json
import os
import matplotlib.pyplot as plt
import logging
from typing import List, Dict

# Import LLM class and our custom modules
from llm import LLM
from knowledge_graph import KnowledgeGraph
from reasoner import ToG2Reasoner
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load custom CSS
def load_css():
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Thiết lập trang
st.set_page_config(
    page_title="Think-on-Graph 2.0 Demo",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
load_css()

# Cache model sentence transformer để tính cosine similarity
@st.cache_resource
def load_model():
    logger.info("Loading SentenceTransformer model (should happen only once)")
    return SentenceTransformer('all-MiniLM-L6-v2')

# Khởi tạo model
model = load_model()

def load_sample_data(data_dir='data'):
    """Load sample data if no files are uploaded"""
    kg = KnowledgeGraph(model=model)
    kg.load_from_files(os.path.join(data_dir, 'entities.json'), os.path.join(data_dir, 'relations.json'), os.path.join(data_dir, 'triples.json'), os.path.join(data_dir, 'documents.json'))    
    return kg

def load_sample_queries(data_dir='data'):
    """Load sample queries from JSON file"""
    try:
        with open(os.path.join(data_dir, 'sample_queries.json'), 'r') as f:
            return json.load(f)['queries']
    except FileNotFoundError:
        return []

def main():
    # Sidebar configuration
    with st.sidebar:
        st.markdown("## Cấu hình")
        
        # Data configuration
        st.markdown("### Dữ liệu")
        use_sample_data = st.checkbox("Sử dụng dữ liệu mẫu", value=True)
        
        # Main content
        if use_sample_data:
            kg = load_sample_data()
            st.sidebar.success("✅ Đã tải dữ liệu mẫu")
        else:
            st.sidebar.markdown("### Tải lên dữ liệu của bạn")
            entities_file = st.sidebar.file_uploader("Tải lên entities.json", type=["json"])
            relations_file = st.sidebar.file_uploader("Tải lên relations.json", type=["json"])
            triples_file = st.sidebar.file_uploader("Tải lên triples.json", type=["json"])
            documents_file = st.sidebar.file_uploader("Tải lên documents.json", type=["json"])
            
            if entities_file and relations_file and triples_file and documents_file:
                # Save uploaded files
                os.makedirs('uploads', exist_ok=True)
                
                with open('uploads/entities.json', 'wb') as f:
                    f.write(entities_file.getbuffer())
                
                with open('uploads/relations.json', 'wb') as f:
                    f.write(relations_file.getbuffer())
                
                with open('uploads/triples.json', 'wb') as f:
                    f.write(triples_file.getbuffer())
                
                with open('uploads/documents.json', 'wb') as f:
                    f.write(documents_file.getbuffer())
                
                # Load data into KG
                kg = KnowledgeGraph(model=model)
                kg.load_from_files('uploads/entities.json', 'uploads/relations.json', 'uploads/triples.json', 'uploads/documents.json')
                st.sidebar.success("✅ Dữ liệu đã được tải lên thành công!")
            else:
                st.warning("Vui lòng tải lên đầy đủ các file dữ liệu cần thiết")
                return
        
        # LLM configuration
        st.markdown("### Cấu hình LLM")
        use_llm = st.checkbox("Sử dụng LLM cho suy luận", value=False)
        
        if use_llm:
            llm_provider = st.selectbox(
                "LLM Provider",
                ["openai", "ollama"],
                help="Chọn provider LLM để sử dụng"
            )
            
            model = st.selectbox(
                "Model",
                ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"] if llm_provider == "openai" else ["llama3", "mistral"],
                help="Chọn model để sử dụng"
            )
            
            use_embedding_api = st.checkbox(
                "Sử dụng Embedding API",
                value=True,
                help="Sử dụng OpenAI/Ollama embedding API để tăng tốc độ tính toán similarity"
            )
            
            api_key = st.text_input(
                "API Key",
                type="password",
                help="Nhập API key của bạn (chỉ cần thiết cho OpenAI)"
            )
            
            api_base = st.text_input(
                "API Base URL (tùy chọn)",
                help="URL cơ sở tùy chỉnh cho API (không bắt buộc)"
            )
            
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                help="Điều chỉnh độ ngẫu nhiên của câu trả lời"
            )
        
        # ToG-2 Parameters
        st.markdown("### Tham số ToG-2")
        max_depth = st.slider(
            "Độ sâu tìm kiếm tối đa",
            min_value=1,
            max_value=5,
            value=3,
            help="Số lượng vòng lặp tối đa trong quá trình tìm kiếm"
        )
        
        max_width = st.slider(
            "Độ rộng tìm kiếm tối đa",
            min_value=1,
            max_value=5,
            value=3,
            help="Số lượng thực thể được giữ lại trong mỗi vòng lặp"
        )
    
    # Initialize LLM instance
    llm_instance = None
    if use_llm:
        if llm_provider == "openai" and not api_key:
            st.error("⚠️ Vui lòng nhập API Key để sử dụng tính năng LLM")
            return
            
        try:
            llm_instance = LLM(
                provider=llm_provider,
                model=model,
                api_key=api_key,
                api_base=api_base if api_base else None,
                temperature=temperature,
                use_embedding_api=use_embedding_api
            )
        except ValueError as e:
            if "API key is required" in str(e):
                st.error("⚠️ Vui lòng nhập API Key để sử dụng tính năng LLM")
                return
            else:
                st.error(f"⚠️ Có lỗi xảy ra: {str(e)}")
                return
                    
    # Initialize reasoner with configured parameters
    reasoner = ToG2Reasoner(kg, llm=llm_instance, max_depth=max_depth, max_width=max_width)
    
    # Load sample queries
    sample_queries = load_sample_queries()
    
    # Add title and description
    st.markdown("""
        <div style='text-align: center; padding: 1rem 0;'>
            <h1>Think-on-Graph 2.0 Demo</h1>
            <p style='font-size: 1.2rem; color: #666;'>
                Kết hợp Knowledge Graph và LLM cho suy luận đa bước sâu và chính xác
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different sections
    tab1, tab2 = st.tabs(["Demo", "Giới thiệu"])
    
    with tab2:
        st.markdown("""
        ## Think-on-Graph 2.0
        
        Think-on-Graph 2.0 (ToG-2) là một cách tiếp cận mới kết hợp chặt chẽ giữa Knowledge Graph và tài liệu văn bản để nâng cao khả năng suy luận đa bước của Large Language Models (LLM).
        
        ### Điểm nổi bật của ToG-2:
        
        1. **Tích hợp chặt chẽ KG và văn bản**: Sử dụng KG để dẫn dắt việc tìm kiếm thông tin trong văn bản, đồng thời sử dụng văn bản để làm phong phú và tinh chỉnh việc tìm kiếm trên đồ thị.
        
        2. **Tìm kiếm đa bước sâu hơn**: Cho phép LLM thực hiện suy luận sâu hơn bằng cách lặp lại quá trình tìm kiếm và tích lũy thông tin.
        
        3. **Suy luận có cơ sở**: Tăng cường độ tin cậy của câu trả lời từ LLM bằng cách cung cấp thông tin được tìm thấy từ cả KG và văn bản.
        
        4. **Linh hoạt**: Có thể áp dụng cho nhiều loại LLM khác nhau mà không cần đào tạo lại.
        
        ### Cách thức hoạt động:
        
        1. **Khởi tạo**: Xác định các thực thể chủ đề từ câu hỏi.
        
        2. **Tìm kiếm đồ thị dựa trên tri thức**: Khám phá các mối quan hệ và thực thể liên quan trên đồ thị kiến thức.
        
        3. **Tìm kiếm văn bản theo hướng dẫn tri thức**: Truy xuất thông tin từ tài liệu dựa trên các thực thể đã khám phá.
        
        4. **Tinh chỉnh thực thể dựa trên ngữ cảnh**: Sử dụng thông tin trong văn bản để xác định thực thể nào là quan trọng nhất.
        
        5. **Lặp lại quá trình**: Tiếp tục khám phá sâu hơn cho đến khi thu thập đủ thông tin.
        
        6. **Đưa ra câu trả lời**: Sinh câu trả lời dựa trên tất cả thông tin thu thập được.
        """)
    
    with tab1:
        # Create columns for the query input section
        query_col1, query_col2 = st.columns([3, 1])
        
        with query_col2:
            if sample_queries:
                # Add a dropdown for sample queries
                selected_query = st.selectbox(
                    "Hoặc chọn câu hỏi mẫu",
                    options=[""] + [q["text"] for q in sample_queries],
                    format_func=lambda x: "Chọn câu hỏi mẫu..." if x == "" else x,
                    help="Chọn một trong những câu hỏi mẫu có sẵn"
                )
        
        with query_col1:
            # Text input for query
            query = st.text_input(
                "Nhập câu hỏi của bạn",
                value=selected_query,
                placeholder="Ví dụ: Nguyễn Nhật Ánh đã viết những tác phẩm nổi tiếng nào?",
                help="Nhập câu hỏi liên quan đến tri thức trong hệ thống"
            )
        
        # Process query
        if query:
            # Execute reasoning with real-time updates
            with st.spinner("Đang thực hiện suy luận..."):
                # Execute reasoning
                reasoning_steps, answer = reasoner.iterative_reasoning(query)
                
                # Sau khi có reasoning_steps, tính toán metrics
                entity_count = 0
                reasoning_steps_count = 0

                # Lấy metadata từ reasoning_steps nếu có
                for step in reasoning_steps:
                    if 'step' in step and step['step'] == 'Process Metadata' and 'metadata' in step:
                        metadata = step['metadata']
                        entity_count = metadata.get('total_entities_visited', 0)
                        break

                # Đếm tổng số bước suy luận thực tế
                for step in reasoning_steps:
                    # Chỉ đếm các bước thực sự là bước suy luận
                    if 'step' in step:
                        # Bỏ qua các bước metadata và tổng thời gian xử lý
                        if step['step'] not in ['Process Metadata', 'Total Processing']:
                            reasoning_steps_count += 1

                # Nếu không tìm thấy metadata về số thực thể
                if entity_count == 0:
                    for step in reasoning_steps:
                        if 'visited_count' in step:
                            entity_count = max(entity_count, step['visited_count'])

                # Hiển thị metrics trước khi tạo tabs
                col1, col2, col3, col4, col5 = st.columns(5)

                with col1:
                    st.markdown(f"""
                        <div class='metric-container'>
                            <div class='metric-value'>{max_width}</div>
                            <div class='metric-label'>Độ rộng tìm kiếm</div>
                        </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                        <div class='metric-container'>
                            <div class='metric-value'>{max_depth}</div>
                            <div class='metric-label'>Độ sâu tìm kiếm</div>
                        </div>
                    """, unsafe_allow_html=True)

                with col3:
                    st.markdown(f"""
                        <div class='metric-container'>
                            <div class='metric-value'>{entity_count}</div>
                            <div class='metric-label'>Thực thể đã duyệt</div>
                        </div>
                    """, unsafe_allow_html=True)

                with col4:
                    st.markdown(f"""
                        <div class='metric-container'>
                            <div class='metric-value'>{reasoning_steps_count}</div>
                            <div class='metric-label'>Bước suy luận</div>
                        </div>
                    """, unsafe_allow_html=True)

                with col5:
                    st.markdown(f"""
                        <div class='metric-container'>
                            <div class='metric-value'>ToG-2</div>
                            <div class='metric-label'>Phương pháp</div>
                        </div>
                    """, unsafe_allow_html=True)

                # Tạo tabs cho kết quả và quá trình suy luận
                result_tab, process_tab, viz_tab = st.tabs(["Kết quả", "Quá trình suy luận", "Trực quan hóa"])
                    
                with result_tab:
                    result_container = st.empty()
            
                with process_tab:
                    reasoning_container = st.container()
                    steps_placeholder = st.empty()
            
                with viz_tab:
                    viz_container = st.empty()
                
                # Function to update reasoning steps in UI
                def update_reasoning_steps(steps):
                    with steps_placeholder.container():
                        for i, step in enumerate(steps, 1):
                            if 'step' in step:
                                # Bỏ qua bước Process Metadata - chỉ dùng để lưu trữ dữ liệu
                                if step['step'] == 'Process Metadata':
                                    continue
                                    
                                # Tạo tiêu đề với thông tin bổ sung nếu có
                                step_title = f"Bước {i}: {step['step']}"
                                if 'visited_count' in step:
                                    step_title += f" ({step['visited_count']} thực thể đã duyệt)"
                                    
                                with st.expander(step_title, expanded=True):
                                    st.markdown(step['description'])
                                    
                                    # Display time if present
                                    if 'time' in step:
                                        st.markdown(f"**Thời gian thực thi:** {step['time']}")
                                    
                                    # Display entities if present
                                    if 'entities' in step:
                                        st.markdown("**Entities:**")
                                        if isinstance(step['entities'], list) and len(step['entities']) > 0:
                                            if isinstance(step['entities'][0], str):
                                                entity_names = [kg.entities[eid]['name'] for eid in step['entities'] if eid in kg.entities]
                                                st.write(", ".join(entity_names))
                                    
                                    # Display topic entities if present
                                    if 'topic_entities' in step:
                                        st.markdown("**Topic Entities:**")
                                        entity_names = [kg.entities[eid]['name'] for eid in step['topic_entities'] if eid in kg.entities]
                                        st.write(", ".join(entity_names))
                                    
                                    # Display entity chains if present
                                    if 'entity_chains' in step:
                                        st.markdown("**Entity chains explored:**")
                                        for chain in step['entity_chains'][:5]:  # Show top 5 chains
                                            st.write(f"- {chain[0]} → {chain[1]} → {chain[2]}")
                                    
                                    # Display documents if present
                                    if 'documents' in step:
                                        st.markdown("**Top relevant sentences:**")
                                        sorted_docs = sorted(step['documents'], key=lambda x: x['score'], reverse=True)
                                        for doc in sorted_docs[:5]:  # Show top 5 docs
                                            st.write(f"- {doc['text']} (From: {doc['entity_name']}, Score: {doc['score']:.2f})")
                                            
                                    # Display answer if present
                                    if 'answer' in step:
                                        st.markdown("**Answer:**")
                                        st.write(step['answer'])
                                        
                                    # Hiển thị metadata nếu có
                                    if 'metadata' in step:
                                        st.markdown("**Process Statistics:**")
                                        metadata = step['metadata']
                                        st.write(f"- Total entities visited: {metadata.get('total_entities_visited', 0)}")
                                        st.write(f"- Total iterations: {metadata.get('total_iterations', 0)}")
                                        st.write(f"- Total entity chains: {metadata.get('total_chains', 0)}")
                                        st.write(f"- Total context snippets: {metadata.get('total_contexts', 0)}")
                
                # Update reasoning steps
                update_reasoning_steps(reasoning_steps)
                
                # Get entities to highlight from reasoning steps
                highlight_entities = []
                entity_names = set()
                
                # Extract entity chains from reasoning steps
                all_chains = []
                for step in reasoning_steps:
                    if 'entity_chains' in step:
                        all_chains.extend(step['entity_chains'])
                
                # Get entity names from chains
                if all_chains:
                    for chain in all_chains:
                        entity_names.update([chain[0], chain[2]])
                
                # Map names back to IDs
                for entity_id, entity in kg.entities.items():
                    if entity['name'] in entity_names:
                        highlight_entities.append(entity_id)
                
                # Update visualization tab
                with viz_container:
                    st.markdown("### Đồ thị kiến thức khám phá")
                    st.pyplot(kg.visualize_graph(highlight_entities=highlight_entities))
            
            # Update results tab
            with result_container:
                st.markdown("### Kết quả")
                with st.container():
                    st.markdown(f"""
                        <div class='results-container'>
                            <h3>Câu hỏi:</h3>
                            <p>{query}</p>
                            <h3>Câu trả lời:</h3>
                            <p>{answer}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
if __name__ == "__main__":
    main()