(function () {
  const SVG_BASE = 'class="icon" aria-hidden="true" focusable="false"';

  const ICONS = {
    microphone: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" ' + SVG_BASE + '><path d="M12 1a4 4 0 0 1 4 4v6a4 4 0 0 1-8 0V5a4 4 0 0 1 4-4zm-1.5 14.95A7.002 7.002 0 0 1 5 9H3a9 9 0 0 0 8 8.94V20H8v2h8v-2h-3v-2.06z"/></svg>',
    stop: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" ' + SVG_BASE + '><path d="M6 6h12v12H6z"/></svg>',
    pause: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" ' + SVG_BASE + '><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>',
    clock: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" ' + SVG_BASE + '><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm.5-13H11v6l5.25 3.15.75-1.23-4.5-2.67V7z"/></svg>',
    text: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" ' + SVG_BASE + '><path d="M14 17H4v2h10v-2zm6-8H4v2h16V9zM4 15h16v-2H4v2zM4 5v2h16V5H4z"/></svg>',
    history: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" ' + SVG_BASE + '><path d="M13 3a9 9 0 0 0-9 9H1l3.89 3.89.07.14L9 12H6c0-3.87 3.13-7 7-7s7 3.13 7 7-3.13 7-7 7c-1.93 0-3.68-.79-4.94-2.06l-1.42 1.42A8.954 8.954 0 0 0 13 21a9 9 0 0 0 0-18zm-1 5v5l4.28 2.54.72-1.21-3.5-2.08V8H12z"/></svg>',
    settings: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" ' + SVG_BASE + '><path d="M19.14 12.94c.04-.3.06-.61.06-.94 0-.32-.02-.64-.07-.94l2.03-1.58c.18-.14.23-.41.12-.61l-1.92-3.32c-.12-.22-.37-.29-.59-.22l-2.39.96c-.5-.38-1.03-.7-1.62-.94l-.36-2.54c-.04-.24-.24-.41-.48-.41h-3.84c-.24 0-.43.17-.47.41l-.36 2.54c-.59.24-1.13.57-1.62.94l-2.39-.96c-.22-.08-.47 0-.59.22L2.74 8.87c-.12.21-.08.47.12.61l2.03 1.58c-.05.3-.09.63-.09.94s.02.64.07.94l-2.03 1.58c-.18.14-.23.41-.12.61l1.92 3.32c.12.22.37.29.59.22l2.39-.96c.5.38 1.03.7 1.62.94l.36 2.54c.05.24.24.41.48.41h3.84c.24 0 .44-.17.47-.41l.36-2.54c.59-.24 1.13-.56 1.62-.94l2.39.96c.22.08.47 0 .59-.22l1.92-3.32c.12-.22.07-.47-.12-.61l-2.01-1.58zM12 15.6c-1.98 0-3.6-1.62-3.6-3.6s1.62-3.6 3.6-3.6 3.6 1.62 3.6 3.6-1.62 3.6-3.6 3.6z"/></svg>',
    upload: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" ' + SVG_BASE + '><path d="M9 16h6v-6h4l-7-7-7 7h4v6zm-4 2h14v2H5v-2z"/></svg>',
    folderOpen: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" ' + SVG_BASE + '><path d="M10 4l2 2h8a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h6zm-6 6h16V8H12.17L10 6H4v4z"/></svg>',
    users: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" ' + SVG_BASE + '><path d="M16 11c1.66 0 3-1.34 3-3s-1.34-3-3-3-3 1.34-3 3 1.34 3 3 3zm-8 0c1.66 0 3-1.34 3-3S9.66 5 8 5 5 6.34 5 8s1.34 3 3 3zm0 2c-2.67 0-8 1.34-8 4v3h10v-3c0-.95.21-1.84.58-2.64C9.5 13.44 8.03 13 8 13zm8 0c-.33 0-.72.03-1.14.08.73.88 1.14 2 1.14 3.17v2.75H24v-3c0-2.66-5.33-4-8-4z"/></svg>',
    warning: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" ' + SVG_BASE + '><path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z"/></svg>',
    brain: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" ' + SVG_BASE + '><path d="M21 11.18V10c0-1.1-.9-2-2-2h-1V6c0-1.1-.9-2-2-2H4c-1.1 0-2 .9-2 2v13c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2v-2h1c1.1 0 2-.9 2-2v-1.82c.58-.28 1-.88 1-1.55 0-.67-.42-1.27-1-1.45zM16 19H4V6h12v13z"/></svg>',
    sparkles: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" ' + SVG_BASE + '><path d="M11 2l1.6 5.4L18 9l-5.4 1.6L11 16l-1.6-5.4L4 9l5.4-1.6L11 2zm7 9l.9 2.7L21 14l-2.1.3L18 17l-.9-2.7L15 14l2.1-.3L18 11zm-13 3l.9 2.7L8 17l-2.1.3L5 20l-.9-2.7L2 17l2.1-.3L5 14z"/></svg>',
    cpu: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" ' + SVG_BASE + '><path d="M9 3H7v2H5v2H3v2h2v2H3v2h2v2H3v2h2v2h2v2h2v-2h2v2h2v-2h2v2h2v-2h2v-2h2v-2h-2v-2h2v-2h-2V9h2V7h-2V5h-2V3h-2v2h-2V3H9zm0 4h6v8H9V7z"/></svg>',
    chartLine: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" ' + SVG_BASE + '><path d="M4 19h16v2H2V3h2v16zm2-4 3-3 3 2 5-6 1.5 1.2-6.5 7.8-3-2L7 17l-1-2z"/></svg>',
    moon: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" ' + SVG_BASE + '><path d="M12.74 2a9 9 0 1 0 9.26 11.1A8 8 0 0 1 12.74 2z"/></svg>',
    sun: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" ' + SVG_BASE + '><path d="M12 4.5A7.5 7.5 0 1 0 19.5 12 7.51 7.51 0 0 0 12 4.5zm0-2h1V5h-2V2.5h1zm0 16.5h1V22h-2v-3h1zM4.93 4.93l1.41 1.41-1.77 1.77L3.16 6.7l1.77-1.77zM17.43 17.43l1.41 1.41-1.77 1.77-1.41-1.41 1.77-1.77zM2 11h3v2H2v-2zm17 0h3v2h-3v-2zm-1.16-5.66 1.77 1.77-1.41 1.41-1.77-1.77 1.41-1.41zM5.57 17.43l1.77 1.77-1.41 1.41-1.77-1.77 1.41-1.41z"/></svg>',
    play: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" ' + SVG_BASE + '><path d="M8 5v14l11-7z"/></svg>',
    fastForward: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" ' + SVG_BASE + '><path d="M4 6l8 6-8 6V6zm8 0 8 6-8 6V6z"/></svg>',
    trash: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" ' + SVG_BASE + '><path d="M9 3h6l1 2h5v2H3V5h5l1-2zm1 6h2v9h-2V9zm4 0h2v9h-2V9zM7 8h10l-1 13H8L7 8z"/></svg>',
    copy: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" ' + SVG_BASE + '><path d="M16 1H6a2 2 0 0 0-2 2v12h2V3h10V1zm3 4H10a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h9a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2zm0 16H10V7h9v14z"/></svg>',
    fileLines: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" ' + SVG_BASE + '><path d="M6 4h9l5 5v11a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2zm8 1.5V10h4.5L14 5.5zM7 12h10v2H7v-2zm0 4h10v2H7v-2zm0-8h5v2H7V8z"/></svg>',
    code: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" ' + SVG_BASE + '><path d="M8.7 16.3 4.4 12l4.3-4.3 1.4 1.4L7.2 12l2.9 2.9-1.4 1.4zm6.6 0-1.4-1.4L17.8 12l-3.9-3.9 1.4-1.4 4.3 4.3-4.3 4.3z"/></svg>',
    robot: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" ' + SVG_BASE + '><path d="M9 2h6v2h3a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2h-1v2H7v-2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h3V2zm0 6a1 1 0 1 0 0 2 1 1 0 0 0 0-2zm6 0a1 1 0 1 0 0 2 1 1 0 0 0 0-2zM9 16h6v2H9v-2z"/></svg>',
    times: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" ' + SVG_BASE + '><path d="M18.3 5.7 12 12l6.3 6.3-1.4 1.4L10.6 13.4 4.3 19.7 2.9 18.3 9.2 12 2.9 5.7 4.3 4.3l6.3 6.3 6.3-6.3 1.4 1.4z"/></svg>',
    listOl: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" ' + SVG_BASE + '><path d="M4 7h2V5H4V4h3v4H4V7zm0 7h2v-1H4v-1h3v4H4v-1h2v-1H4v-0zM10 6h10v2H10V6zm0 6h10v2H10v-2zm0 6h10v2H10v-2z"/></svg>',
    paperPlane: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" ' + SVG_BASE + '><path d="M2 21 23 12 2 3v6l15 3-15 3v6z"/></svg>',
    spinner: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" ' + SVG_BASE + '><path d="M12 2a10 10 0 1 0 10 10h-2a8 8 0 1 1-8-8V2z"/></svg>',
    checkCircle: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" ' + SVG_BASE + '><path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm-1 14.2-4.2-4.2 1.4-1.4 2.8 2.8 6-6 1.4 1.4-7.4 7.4z"/></svg>',
    timesCircle: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" ' + SVG_BASE + '><path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm4.3 13.9-1.4 1.4L12 13.4 8.1 17.3l-1.4-1.4L10.6 12 6.7 8.1l1.4-1.4L12 10.6l3.9-3.9 1.4 1.4L13.4 12l2.9 2.9z"/></svg>',
    infoCircle: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" ' + SVG_BASE + '><path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm1 15h-2v-6h2zm0-8h-2V7h2z"/></svg>',
  };

  const FA_CLASS_TO_ICON = {
    'fa-cloud-arrow-up': 'upload',
    'fa-folder-open': 'folderOpen',
    'fa-microphone': 'microphone',
    'fa-stop': 'stop',
    'fa-pause': 'pause',
    'fa-clock': 'clock',
    'fa-font': 'text',
    'fa-history': 'history',
    'fa-cog': 'settings',
    'fa-gear': 'settings',
    'fa-upload': 'upload',
    'fa-microchip': 'cpu',
    'fa-exclamation-triangle': 'warning',
    'fa-magic': 'sparkles',
    'fa-brain': 'brain',
    'fa-users': 'users',
    'fa-fast-forward': 'fastForward',
    'fa-play': 'play',
    'fa-trash': 'trash',
    'fa-trash-alt': 'trash',
    'fa-copy': 'copy',
    'fa-file-lines': 'fileLines',
    'fa-code': 'code',
    'fa-chart-line': 'chartLine',
    'fa-moon': 'moon',
    'fa-sun': 'sun',
    'fa-robot': 'robot',
    'fa-times': 'times',
    'fa-list-ol': 'listOl',
    'fa-paper-plane': 'paperPlane',
    'fa-circle-notch': 'spinner',
    'fa-check-circle': 'checkCircle',
    'fa-times-circle': 'timesCircle',
    'fa-info-circle': 'infoCircle',
    'fa-user-circle': 'robot',
  };

  function icon(name, extraClass = '') {
    const svg = ICONS[name];
    if (!svg) return '';
    const extra = extraClass ? ` ${extraClass}` : '';
    return svg.replace('class="icon"', `class="icon${extra}"`);
  }

  function iconNameFromElement(element) {
    const classes = Array.from(element.classList);
    for (const className of classes) {
      if (FA_CLASS_TO_ICON[className]) return FA_CLASS_TO_ICON[className];
    }
    return null;
  }

  function convertIcons(root = document) {
    const nodes = [];
    if (root.matches && root.matches('i[class*="fa"], span[class*="fa"]')) {
      nodes.push(root);
    }
    if (root.querySelectorAll) {
      nodes.push(...root.querySelectorAll('i[class*="fa"], span[class*="fa"]'));
    }

    nodes.forEach((node) => {
      const iconName = iconNameFromElement(node);
      if (!iconName) return;
      const extraClass = Array.from(node.classList)
        .filter((className) => !className.startsWith('fa-') && className !== 'fas' && className !== 'far' && className !== 'fab' && className !== 'fa-solid' && className !== 'fa-regular' && className !== 'fa-brands')
        .join(' ');
      node.outerHTML = icon(iconName, extraClass);
    });
  }

  function injectStyles() {
    if (document.getElementById('icon-fallback-styles')) return;
    const style = document.createElement('style');
    style.id = 'icon-fallback-styles';
    style.textContent = `
      .icon {
        display: inline-block;
        width: 1em;
        height: 1em;
        vertical-align: -0.125em;
        fill: currentColor;
        flex: 0 0 auto;
      }
      .icon.fa-spin {
        animation: icon-spin 1s linear infinite;
        transform-origin: center;
      }
      @keyframes icon-spin {
        to { transform: rotate(360deg); }
      }
    `;
    document.head.appendChild(style);
  }

  function boot() {
    injectStyles();
    convertIcons(document);

    if ('MutationObserver' in window && document.body) {
      const observer = new MutationObserver((mutations) => {
        for (const mutation of mutations) {
          mutation.addedNodes.forEach((node) => {
            if (node.nodeType !== Node.ELEMENT_NODE) return;
            convertIcons(node);
          });
        }
      });
      observer.observe(document.body, { childList: true, subtree: true });
    }
  }

  window.icon = icon;
  window.replaceFontAwesomeIcons = convertIcons;

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot, { once: true });
  } else {
    boot();
  }
})();