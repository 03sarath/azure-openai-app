from flask import Blueprint, render_template, request, jsonify, session
from app.models.qa_model import QAModel

bp = Blueprint('main', __name__)
qa_model = QAModel()

@bp.route('/')
def index():
    """Render the main chat interface."""
    return render_template('index.html')

@bp.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests."""
    data = request.get_json()
    question = data.get('question')
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    # Get chat history from session
    chat_history = session.get('chat_history', [])
    
    try:
        # Get answer from QA model
        result = qa_model.get_answer(question, chat_history)
        
        # Update chat history
        chat_history.append((question, result['answer']))
        session['chat_history'] = chat_history
        
        return jsonify({
            'answer': result['answer'],
            'sources': [doc.page_content for doc in result['sources']]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/clear', methods=['POST'])
def clear_chat():
    """Clear the chat history."""
    session['chat_history'] = []
    return jsonify({'status': 'success'}) 