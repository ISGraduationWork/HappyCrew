// モーダルウィンドウとボタン、閉じるボタンの要素を取得
var modal = document.querySelector('.modal');           // モーダルウィンドウの要素
var btn = document.querySelector('.add-class-btn');     // モーダルを開くボタン
var close = document.querySelector('.close');           // モーダルを閉じるボタン

// 「追加」ボタンがクリックされたとき、モーダルを表示するイベントリスナーを設定
btn.addEventListener("click", function(){
  modal.style.display = "block";    // モーダルの表示状態を「ブロック」に設定して表示
});

// 「閉じる」ボタンがクリックされたとき、モーダルを非表示にするイベントリスナーを設定
close.addEventListener("click", function(){
  modal.style.display = "none";     // モーダルの表示状態を「none」に設定して非表示
});

// モーダル外がクリックされたとき、モーダルを非表示にするイベントリスナーを設定
window.addEventListener("click", function(e){
  if (e.target == modal) {          // クリック対象がモーダル自身の場合
    modal.style.display = "none";    // モーダルを非表示
  }
});

// クラス作成フォームの要素を取得
var form = document.querySelector('.create-class-form');

// フォーム送信時のイベントハンドラを設定
form.onsubmit = function (event) {
  event.preventDefault();    // デフォルトの送信動作を防止

  // クラス名と担当教師名の入力値を取得
  var class_name = document.getElementById('class_name').value;
  var teacher = document.getElementById('class_teacher').value;

  // 入力チェック：クラス名と担当教師名の入力が必須
  if (!class_name || !teacher) {
    alert('クラス名と担当教師を入力してください。');
    return;   // 入力が足りない場合は処理を中断
  }

  // 非同期通信でクラス作成リクエストを送信
  fetch('/create_class', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',    // JSON形式のデータを送信
    },
    body: JSON.stringify({                   // リクエストボディにクラス情報をJSON形式で送信
      class_name: class_name,
      teacher: teacher,
    }),
  })
    .then(function (response) {
      if (response.ok) {
        return response.json();              // 成功時はレスポンスをJSONで取得
      } else {
        throw new Error('クラスの作成に失敗しました');  // エラーハンドリング
      }
    })
    .then(function (data) {
      alert(data.message);                   // クラス作成成功メッセージを表示
      modal.style.display = 'none';          // モーダルを非表示
      window.location.reload();              // ページをリロードして変更を反映
    })
    .catch(function (error) {
      console.error('Error:', error);        // コンソールにエラーログを出力
      alert('エラーが発生しました: ' + error.message);  // ユーザーにエラーメッセージを表示
    });
};
