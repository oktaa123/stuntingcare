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

    const showToast = (message, type = "success") => {
        const stack = document.querySelector("[data-toast-stack]");
        if (!stack) return;

        const toast = document.createElement("div");
        toast.className = `app-toast${type === "error" ? " is-error" : ""}`;
        toast.setAttribute("role", type === "error" ? "alert" : "status");
        toast.innerHTML = `<i class="bi ${type === "error" ? "bi-exclamation-circle-fill" : "bi-check-circle-fill"}" aria-hidden="true"></i><span>${message}</span>`;
        stack.appendChild(toast);
        requestAnimationFrame(() => toast.classList.add("is-visible"));
        window.setTimeout(() => {
            toast.classList.remove("is-visible");
            window.setTimeout(() => toast.remove(), 220);
        }, 4200);
    };

    const form = document.querySelector("[data-prediction-form]");
    const analysisButton = form?.querySelector("[data-analysis-button]");
    const resetAnalysisButton = () => {
        if (!analysisButton) return;
        analysisButton.disabled = false;
        analysisButton.classList.remove("is-loading");
        analysisButton.innerHTML = '<span data-button-label>Analisis Sekarang</span><i class="bi bi-arrow-right" data-button-icon aria-hidden="true"></i>';
        document.body.classList.remove("is-leaving");
    };

    if (form) {
        form.addEventListener("submit", (event) => {
            if (!form.checkValidity()) {
                event.preventDefault();
                form.classList.add("was-validated");
                form.querySelector(":invalid")?.focus();
                return;
            }

            analysisButton.disabled = true;
            analysisButton.classList.add("is-loading");
            analysisButton.innerHTML = '<span>Memproses skrining...</span><i class="bi bi-arrow-repeat" aria-hidden="true"></i>';
            document.body.classList.add("is-leaving");
        });

        // Browsers may restore a page from the back/forward cache. The form must
        // remain usable and may never retain a stale loading state.
        window.addEventListener("pageshow", resetAnalysisButton);
    }

    const downloadButton = document.querySelector("[data-pdf-download]");
    if (downloadButton) {
        downloadButton.addEventListener("click", async (event) => {
            event.preventDefault();
            if (downloadButton.classList.contains("is-loading")) return;

            const originalMarkup = downloadButton.innerHTML;
            downloadButton.classList.add("is-loading");
            downloadButton.setAttribute("aria-disabled", "true");
            downloadButton.innerHTML = '<i class="bi bi-arrow-repeat" aria-hidden="true"></i><span>Menyiapkan PDF...</span>';

            try {
                const response = await fetch(downloadButton.href, {
                    credentials: "same-origin",
                    headers: { Accept: "application/pdf" },
                });
                const contentType = response.headers.get("Content-Type") || "";
                if (!response.ok || !contentType.includes("application/pdf")) {
                    throw new Error("PDF tidak tersedia");
                }

                const blob = await response.blob();
                const disposition = response.headers.get("Content-Disposition") || "";
                const filenameMatch = disposition.match(/filename\*?=(?:UTF-8''|\")?([^\";]+)/i);
                const filename = filenameMatch
                    ? decodeURIComponent(filenameMatch[1].trim())
                    : "StuntingCare-hasil-skrining.pdf";
                const objectUrl = URL.createObjectURL(blob);
                const temporaryLink = document.createElement("a");
                temporaryLink.href = objectUrl;
                temporaryLink.download = filename;
                document.body.appendChild(temporaryLink);
                temporaryLink.click();
                temporaryLink.remove();
                window.setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
                showToast("PDF berhasil dibuat dan diunduh.");
            } catch (error) {
                showToast("PDF belum berhasil diunduh. Silakan coba kembali.", "error");
            } finally {
                downloadButton.classList.remove("is-loading");
                downloadButton.removeAttribute("aria-disabled");
                downloadButton.innerHTML = originalMarkup;
            }
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
