document.addEventListener('DOMContentLoaded', function () {
    var toggleTitles = document.querySelectorAll('.toggle-title');

    toggleTitles.forEach(function(toggleTitle) {
        toggleTitle.addEventListener('click', function () {
            var toggleContent = toggleTitle.nextElementSibling;

            if (toggleContent.style.maxHeight) {
                toggleContent.style.maxHeight = null;
                toggleTitle.querySelector('.toggle-arrow').textContent = '🡆';
            } else {
                toggleContent.style.maxHeight = toggleContent.scrollHeight + 'px';
                toggleTitle.querySelector('.toggle-arrow').textContent = '🡇';
            }
        });

        // Afficher le contenu par défaut au chargement de la page
        var toggleContent = toggleTitle.nextElementSibling;
        toggleContent.style.maxHeight = toggleContent.scrollHeight + 'px';
        toggleTitle.querySelector('.toggle-arrow').textContent = '🡇';
    });
});