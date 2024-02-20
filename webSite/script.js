document.addEventListener('DOMContentLoaded', function () {
    var toggleTitles = document.querySelectorAll('.toggle-title');

    toggleTitles.forEach(function(toggleTitle) {
        toggleTitle.addEventListener('click', function () {
            var toggleContent = toggleTitle.nextElementSibling;

            if (toggleContent.style.maxHeight) {
                toggleContent.style.maxHeight = null;
                toggleContent.style.overflow = 'hidden'; // Masquer le contenu qui dépasse lors de la fermeture
                toggleTitle.querySelector('.toggle-arrow').textContent = '🡆';
            } else {
                toggleContent.style.maxHeight = toggleContent.scrollHeight + 'px';
                toggleContent.style.overflow = 'visible'; // Utiliser 'visible' lorsque le contenu est ouvert
                toggleTitle.querySelector('.toggle-arrow').textContent = '🡇';
            }
        });

        // Afficher le contenu par défaut au chargement de la page
        var toggleContent = toggleTitle.nextElementSibling;
        toggleContent.style.maxHeight = toggleContent.scrollHeight + 'px';
        toggleContent.style.overflow = 'visible'; // Assurez-vous que le défilement est activé par défaut
        toggleTitle.querySelector('.toggle-arrow').textContent = '🡇';

        // Détecter la fin de l'animation de fermeture et modifier le mode d'overflow
        toggleContent.addEventListener('transitionend', function() {
            if (!toggleContent.style.maxHeight) {
                toggleContent.style.overflow = 'hidden'; // Rétablir le mode d'overflow à 'hidden' lorsque l'animation est terminée
            }
        });
    });
});
