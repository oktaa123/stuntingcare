document.addEventListener("DOMContentLoaded", () => {
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const revealItems = document.querySelectorAll(".reveal");

    if (reducedMotion || !("IntersectionObserver" in window)) {
        revealItems.forEach((item) => item.classList.add("is-visible"));
    } else {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach((entry) => {
                if (!entry.isIntersecting) return;
                entry.target.classList.add("is-visible");
                observer.unobserve(entry.target);
            });
        }, { threshold: 0.08, rootMargin: "0px 0px -25px" });
        revealItems.forEach((item) => observer.observe(item));
    }

    document.querySelectorAll("[data-progress]").forEach((bar) => {
        const value = Math.max(0, Math.min(100, Number(bar.dataset.progress) || 0));
        requestAnimationFrame(() => { bar.style.width = `${value}%`; });
    });

    const form = document.querySelector("[data-prediction-form]");
    if (form) {
        form.addEventListener("submit", (event) => {
            if (!form.checkValidity()) {
                event.preventDefault();
                form.classList.add("was-validated");
                const firstInvalidField = form.querySelector(":invalid");
                firstInvalidField?.focus();
                return;
            }
            const button = form.querySelector("button[type='submit']");
            button.disabled = true;
            button.innerHTML = '<span>Menganalisis...</span><i class="bi bi-arrow-repeat spin"></i>';
        });
    }

    const nav = document.querySelector("#mainNav");
    document.querySelectorAll("#mainNav .nav-link").forEach((link) => {
        link.addEventListener("click", () => {
            if (!nav?.classList.contains("show") || typeof bootstrap === "undefined") return;
            bootstrap.Collapse.getOrCreateInstance(nav).hide();
        });
    });
});
