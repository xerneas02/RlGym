function toggleMenu(menuIndex) {
    var contentId = "content" + menuIndex;
    var content = document.getElementById(contentId);
    var toggleArrow = document.querySelectorAll('.toggle-arrow')[menuIndex - 1];
  
    content.classList.toggle('show');
  
    // Mettre à jour le caractère de la flèche
    toggleArrow.innerHTML = content.classList.contains('show') ? '🡇' : '🡆';
  
    // Forcer le recalcul de la hauteur maximale pour déclencher l'animation de fermeture
    if (!content.classList.contains('show')) {
      content.style.maxHeight = "0";
    } else {
      // Utiliser la propriété scrollHeight pour définir la hauteur maximale
      content.style.maxHeight = content.scrollHeight + "px";
    }
  }
  
