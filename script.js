function sendMessage() {
    const inputElement = document.getElementById('message-input');
    const message = inputElement.value.trim();
  
    if (message === '') return;
  
    const chatMessagesElement = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message sent';
    messageDiv.innerText = message;
    chatMessagesElement.appendChild(messageDiv);
  
    inputElement.value = '';
    chatMessagesElement.scrollTop = chatMessagesElement.scrollHeight;
  }
