function redirectToPage2() {
    window.location.href = "/page2"; // Chuyển đến page2
}

function redirectToPage1() {
    window.location.href = "/page1"; // Chuyển đến page1
}





document.addEventListener('keydown', function (event) {
    const inputs = document.querySelectorAll('.input input[type="number"]');
    const inputArray = Array.from(inputs);
    const currentIndex = inputArray.indexOf(document.activeElement);
    
    const totalRows = inputs.length > 36 ? 6 : 4; // Xác định số lượng ô input trong trang hiện tại
    
    let nextIndex;

    switch (event.key) {
        case 'ArrowUp':
            nextIndex = currentIndex - totalRows; // Lên trên cùng một cột
            break;
        case 'ArrowDown':
            nextIndex = currentIndex + totalRows; // Xuống dưới cùng một cột
            break;
        case 'ArrowLeft':
            nextIndex = currentIndex - 1; // Sang trái
            break;
        case 'ArrowRight':
            nextIndex = currentIndex + 1; // Sang phải
            break;
        default:
            return; // Nếu không phải các phím điều hướng, thoát khỏi hàm
    }

    if (nextIndex >= 0 && nextIndex < inputArray.length) {
        inputArray[nextIndex].focus();
    }

    // Ngăn việc di chuyển con trỏ trong input khi nhấn phím điều hướng
    event.preventDefault();
});




function toggleCheckbox(selected) {
    const natureCheckbox = document.getElementById('nature_checkbox');
    const socialCheckbox = document.getElementById('social_checkbox');

    // Nếu "Tự nhiên" được chọn, bỏ chọn "Xã hội"
    if (selected === 'nature') {
        if (natureCheckbox.checked) {
            socialCheckbox.checked = false;
        }
    }

    // Nếu "Xã hội" được chọn, bỏ chọn "Tự nhiên"
    if (selected === 'social') {
        if (socialCheckbox.checked) {
            natureCheckbox.checked = false;
        }
    }
}





document.addEventListener('keydown', function(event) {
    if (event.key === 'Enter') {
        event.preventDefault(); // Ngăn chặn hành vi mặc định của phím Enter (nếu có)
        document.querySelector('.btn-view-results').click(); // Giả lập một cú nhấp chuột vào nút
    }
});






function showImage(n) {
    let dropdown, image;

    if (n == 1) {
        dropdown = document.getElementById("imageDropdown1");
        image = document.getElementById("displayedImage1");
    } else if (n == 2) {
        dropdown = document.getElementById("imageDropdown2");
        image = document.getElementById("displayedImage2");
    } else if (n == 3){
        dropdown = document.getElementById("imageDropdown3");
        image = document.getElementById("displayedImage3");
    } else {
        dropdown = document.getElementById("imageDropdown4");
        image = document.getElementById("displayedImage4");
    }
    const selectedValue = dropdown.value;

    if (selectedValue) {
        image.src = selectedValue;
        image.style.display = "block";
    } else {
        image.style.display = "none";
    }
}


function goBack() {
    window.history.back();
}