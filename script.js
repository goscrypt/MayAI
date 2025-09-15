// A simple function to simulate a bot's thinking process and display messages
function displayMessage(sender, text) {
    const chatBox = document.getElementById('chat-box');
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', sender === 'bot' ? 'bot-message' : 'user-message');
    messageDiv.innerHTML = `<div class="content">${text}</div>`;
    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

// Function to handle the file upload and processing
async function processDocument(file) {
    displayMessage('bot', `Processing "${file.name}"... Please wait.`);
    const statusElement = document.getElementById('status');
    statusElement.textContent = `Processing "${file.name}"...`;

    // Initialize our pipeline for generating embeddings
    const pipeline = await transformers.pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

    let textContent = '';
    // Use pdf.js to extract text from a PDF
    if (file.type === 'application/pdf') {
        const url = URL.createObjectURL(file);
        const pdf = await pdfjsLib.getDocument(url).promise;
        for (let i = 1; i <= pdf.numPages; i++) {
            const page = await pdf.getPage(i);
            const text = await page.getTextContent();
            textContent += text.items.map(item => item.str).join(' ');
        }
    } else {
        // For other file types, we can assume plain text for simplicity
        textContent = await file.text();
    }

    // Split text into chunks for better retrieval
    const chunks = textContent.match(/[^.!?]+[.!?]*/g) || [];
    if (chunks.length === 0) {
        displayMessage('bot', 'No readable text found in the document.');
        statusElement.textContent = 'Failed to process document.';
        return;
    }

    // Create a vector database (using IndexedDB in this case)
    const dbName = 'documentStore';
    const dbVersion = 1;
    const request = indexedDB.open(dbName, dbVersion);

    request.onupgradeneeded = (event) => {
        const db = event.target.result;
        db.createObjectStore('chunks', { keyPath: 'id' });
    };

    request.onsuccess = async (event) => {
        const db = event.target.result;
        const transaction = db.transaction('chunks', 'readwrite');
        const store = transaction.objectStore('chunks');

        displayMessage('bot', `Generating embeddings for ${chunks.length} text chunks...`);

        // Generate embeddings for all chunks and store them
        for (let i = 0; i < chunks.length; i++) {
            const output = await pipeline(chunks[i], { pooling: 'mean', normalize: true });
            store.add({ id: i, text: chunks[i], embedding: output.data });
        }

        transaction.oncomplete = () => {
            db.close();
            displayMessage('bot', 'Document processing complete! You can now ask me questions.');
            statusElement.textContent = `"${file.name}" is ready.`;
        };
    };

    request.onerror = () => {
        displayMessage('bot', 'Error: Could not access local database.');
    };

    return pipeline;
}

// Function to handle a user's question
async function handleQuestion(question) {
    displayMessage('user', question);
    
    const dbName = 'documentStore';
    const request = indexedDB.open(dbName, 1);

    request.onsuccess = async (event) => {
        const db = event.target.result;
        const transaction = db.transaction('chunks', 'readonly');
        const store = transaction.objectStore('chunks');
        const allChunks = await store.getAll();
        
        // Find the most relevant chunk
        const pipeline = await transformers.pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
        const questionEmbedding = (await pipeline(question, { pooling: 'mean', normalize: true })).data;
        
        let bestMatch = null;
        let maxSimilarity = -1;

        // Simple cosine similarity search
        for (const chunk of allChunks) {
            let dotProduct = 0;
            for (let i = 0; i < questionEmbedding.length; i++) {
                dotProduct += questionEmbedding[i] * chunk.embedding[i];
            }
            if (dotProduct > maxSimilarity) {
                maxSimilarity = dotProduct;
                bestMatch = chunk.text;
            }
        }

        db.close();

        if (bestMatch && maxSimilarity > 0.5) { // Threshold for relevance
            // Here you would use a Language Model to generate an answer
            // based on the bestMatch. For now, we will just return the most
            // relevant text from the document.
            displayMessage('bot', 'Based on the document, here is the most relevant information:');
            displayMessage('bot', bestMatch);
        } else {
            displayMessage('bot', 'I could not find a relevant answer in the document you provided.');
        }
    };
    
    request.onerror = () => {
        displayMessage('bot', 'Error: Cannot access document database.');
    };
}

// Set up event listeners for the chat interface
document.getElementById('file-input').addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        processDocument(file);
    }
});

document.getElementById('send-button').addEventListener('click', () => {
    const input = document.getElementById('chat-input');
    const question = input.value.trim();
    if (question) {
        handleQuestion(question);
        input.value = '';
    }
});

document.getElementById('chat-input').addEventListener('keypress', (event) => {
    if (event.key === 'Enter') {
        document.getElementById('send-button').click();
    }
});

// Set up Web Speech API for voice input
const voiceButton = document.getElementById('voice-button');
if ('webkitSpeechRecognition' in window) {
    const recognition = new webkitSpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    voiceButton.addEventListener('click', () => {
        recognition.start();
        voiceButton.textContent = '🔴';
    });

    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        document.getElementById('chat-input').value = transcript;
        voiceButton.textContent = '🎤';
        document.getElementById('send-button').click();
    };

    recognition.onend = () => {
        voiceButton.textContent = '🎤';
    };
} else {
    voiceButton.style.display = 'none'; // Hide if not supported
}
