document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('fileInput');
    const fileName = document.getElementById('fileName');
    const submitButton = document.getElementById('submitButton');
    const uploadForm = document.getElementById('uploadForm');
    const loadingElement = document.getElementById('loading');

    // ファイル選択時にファイル名を表示し、送信ボタンを表示
    fileInput.addEventListener('change', function(event) {
        const file = event.target.files[0];
        if (file) {
            fileName.textContent = `選択されたファイル: ${file.name}`;
            submitButton.style.display = 'block';
        }
        });

    // フォーム送信時の処理
    uploadForm.addEventListener('submit', function(event) {
        event.preventDefault();

      // ローディング表示
        loadingElement.style.display = 'block';

        const formData = new FormData();
        const file = fileInput.files[0];
        const classId = document.getElementById('classId').value;

        formData.append('file', file);
        formData.append('class_id', classId);

        fetch('/upload_video', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // ローディング非表示
            loadingElement.style.display = 'none';

            if (data.error) {
            alert(data.error);
            } else {
            if (confirm(data.message)) {
                window.location.reload();
            }
            }
        })
        .catch(error => {
            // ローディング非表示
            loadingElement.style.display = 'none';

            console.error('Error:', error);
            alert('ファイルのアップロード中にエラーが発生しました');
        });
    });
});
