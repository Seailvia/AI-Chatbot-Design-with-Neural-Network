function sendMessage() {
    var userInput = document.getElementById('user-input');
    if (!userInput) {
        console.error('Input field not found');
        return;
    }
    
    var userText = userInput.value.trim();
    if (!userText) {
        console.warn('No message to send');
        return;
    }

    console.log('User input:', userText); // 日志调试

    var chatContent = document.getElementById('chat');
    if (!chatContent) {
        console.error('Chat content area not found');
        return;
    }

    // 添加用户消息
    var userMessage = document.createElement('div');
    userMessage.classList.add('chat-message', 'user-message');
    userMessage.textContent = userText;
    chatContent.appendChild(userMessage);
    userInput.value = ''; // 清空输入框

    // 发出请求
    fetch('/get_response', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userText })
    })
    .then(response => {
        if (!response.ok) throw new Error('Network response was not ok');
        return response.json();
    })
    .then(data => {
        console.log('Response received:', data); // 日志调试
        var botMessage = document.createElement('div');
        botMessage.classList.add('chat-message', 'bot-message');
        botMessage.textContent = data.response;
        chatContent.appendChild(botMessage);
        chatContent.scrollTop = chatContent.scrollHeight; // 滚动到底部
    })
    .catch(error => {
        console.error('Error:', error); // 捕获错误
    });
}
