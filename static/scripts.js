function clearFields() {
    document.querySelector('input[name="question"]').value = '';
    document.querySelector('input[name="reply"]').value = '';
    document.querySelector('input[name="memo_id"]').value = '';
    document.querySelector('select[name="state_name"]').selectedIndex = 0;
    document.getElementById('related-questions-box').innerHTML = ''; // Clear related questions
}



document.addEventListener("DOMContentLoaded", function() {
    console.log("Custom JS is working!");
});
