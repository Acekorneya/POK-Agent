try {
  const theme = localStorage.getItem("pok_theme");
  document.documentElement.dataset.theme = ["teal", "dark", "light"].includes(theme) ? theme : "teal";
} catch {
  document.documentElement.dataset.theme = "teal";
}
