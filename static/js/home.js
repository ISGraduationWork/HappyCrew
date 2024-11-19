document.addEventListener("DOMContentLoaded", function() {
    const grid = document.getElementById("classes-grid");

    // Sortable.jsの初期化
    const sortable = Sortable.create(grid, {
        animation: 150,
        onEnd: function(event) {
            const order = [];
            grid.querySelectorAll(".class-card").forEach((element, index) => {
                order.push(element.getAttribute("data-class-id"));
            });

            // サーバーに新しい順序を送信
            fetch("/update_class_order", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ order: order }),
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    console.log("Class order updated successfully");
                } else {
                    console.error("Failed to update class order");
                }
                location.reload();
            })
            .catch(error => {
                console.error("Error:", error);
            });
        },
    });

    // クラス編集モーダルの初期化
    const editModal = document.getElementById("editModal");
    const deleteModal = document.getElementById("deleteModal");
    const closeBtns = document.querySelectorAll(".close");

    // モーダルを閉じる処理
    closeBtns.forEach(function(span) {
        span.onclick = function() {
            editModal.style.display = "none";
            deleteModal.style.display = "none";
        };
    });

    // 外部クリックでモーダルを閉じる
    window.onclick = function(event) {
        if (event.target == editModal) {
            editModal.style.display = "none";
        }
        if (event.target == deleteModal) {
            deleteModal.style.display = "none";
        }
    };

    // クラス編集ボタンのイベントリスナー
    document.querySelectorAll(".edit-class-btn").forEach(button => {
        button.addEventListener("click", function() {
            const classId = this.getAttribute("data-class-id");
            const className = this.getAttribute("data-class-name");
            const classTeacher = this.getAttribute("data-class-teacher");

            document.getElementById("edit_class_id").value = classId;
            document.getElementById("edit_class_name").value = className;
            document.getElementById("edit_class_teacher").value = classTeacher;

            editModal.style.display = "block"; // モーダルを表示
        });
    });

    // クラス編集フォームの送信処理
    const editForm = document.getElementById("edit-class-form");
    editForm.onsubmit = function(event) {
        event.preventDefault();

        const classId = document.getElementById("edit_class_id").value;
        const className = document.getElementById("edit_class_name").value;
        const classTeacher = document.getElementById("edit_class_teacher").value;

        // サーバーにクラス情報の更新をリクエスト
        fetch("/edit_class", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ class_id: classId, name: className, teacher: classTeacher }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                alert("クラスが更新されました。");
                editModal.style.display = "none"; // モーダルを閉じる
                location.reload(); // ページをリロードして変更を反映
            } else {
                alert("クラスの更新に失敗しました。");
            }
        })
        .catch(error => {
            console.error("Error:", error);
        });
    };

    // クラス削除ボタンのイベントリスナー
    document.querySelectorAll(".delete-class-btn").forEach(button => {
        button.addEventListener("click", function() {
            const classId = this.getAttribute("data-class-id");
            document.getElementById("delete_class_id").value = classId;

            deleteModal.style.display = "block"; // モーダルを表示
        });
    });

    // クラス削除フォームの送信処理
    const deleteForm = document.getElementById("delete-class-form");
    deleteForm.onsubmit = function(event) {
        event.preventDefault();

        const classId = document.getElementById("delete_class_id").value;

        // サーバーに削除リクエストを送信
        fetch("/delete_class", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ class_id: classId }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                alert("クラスが削除されました。");
                deleteModal.style.display = "none"; // モーダルを閉じる
                location.reload(); // ページをリロードして変更を反映
            } else {
                alert("クラスの削除に失敗しました。");
            }
        })
        .catch(error => {
            console.error("Error:", error);
        });
    };
});