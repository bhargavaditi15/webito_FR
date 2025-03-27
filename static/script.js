document.addEventListener("DOMContentLoaded", function () {
    let attendanceForm = document.querySelector("form[action='/mark-attendance']");

    if (attendanceForm) {
        attendanceForm.addEventListener("submit", function (event) {
            event.preventDefault(); // Prevent form from reloading the page

            fetch('/mark-attendance', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    let messageBox = document.getElementById("attendance-success");
                    messageBox.classList.remove("d-none");
                    setTimeout(() => messageBox.classList.add("d-none"), 3000);
                })
                .catch(error => console.error('Error:', error));
        });
    }
});


// Remove flash messages after 3 seconds
let flashMessages = document.querySelectorAll(".flash-message");
flashMessages.forEach(message => {
    setTimeout(() => {
        message.style.opacity = "0";
        setTimeout(() => message.remove(), 500); // Remove element after fade out
    }, 3000);
});

// Form validation for name input
let form = document.querySelector("form");
if (form) {
    form.addEventListener("submit", function (event) {
        let nameInput = document.querySelector("input[name='name']");
        if (nameInput && nameInput.value.trim() === "") {
            event.preventDefault();
            alert("Please enter your name before registering.");
        }
    });
}

// Smooth scrolling for internal links
let links = document.querySelectorAll("a[href^='#']");
links.forEach(link => {
    link.addEventListener("click", function (event) {
        let targetId = this.getAttribute("href");
        let targetElement = document.querySelector(targetId);
        if (targetElement) {
            event.preventDefault();
            targetElement.scrollIntoView({
                behavior: "smooth"
            });
        }
    });
});

// Navbar toggler for mobile
let navbarToggler = document.querySelector(".navbar-toggler");
let navLinks = document.querySelector("#navbarNav");
if (navbarToggler && navLinks) {
    navbarToggler.addEventListener("click", function () {
        navLinks.classList.toggle("show");
    });
}

