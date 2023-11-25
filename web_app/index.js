// function handleFileSelect(event) {
//     // Quando um arquivo é selecionado, envie o formulário para o servidor
//     document.getElementById('uploadForm').submit();
// }

function handleFileSelect(event) {
    // Obtém o arquivo selecionado
    const fileInput = event.target;
    const file = fileInput.files[0];

    // Verifica se um arquivo foi selecionado
    if (file) {
        // Cria um objeto FormData para enviar o arquivo
        const formData = new FormData();
        formData.append('arquivo', file);

        // Opções da solicitação
        const options = {
            method: 'POST',
            body: formData
        };

        // URL do servidor para onde você está enviando a solicitação POST
        const url = '/send_image';

        // Realiza a solicitação POST usando fetch
        fetch(url, options)
        // fetch(url, options)
        //     .then(response => response.json())
        //     .then(data => {
        //         console.log('Resposta do servidor:', data);
        //     })
        //     .catch(error => {
        //         console.error('Erro:', error);
        //     });
        // event.preventDefault();
    }
}