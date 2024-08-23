// Get the sliders
const positionSlider = document.getElementById("position_slider");
const slider2 = document.getElementById("slider2");
// Get the dropdown menus
const dropdown1 = document.getElementById("dropdown1");
const dropdown2 = document.getElementById("dropdown2");
// Add event listeners
positionSlider.addEventListener("input", function () {
    console.log("Position: ", positionSlider.value);
});
slider2.addEventListener("input", function () {
    console.log("Slider 2: ", slider2.value);
});
dropdown1.addEventListener("change", function () {
    console.log("Dropdown 1: ", dropdown1.value);
});
dropdown2.addEventListener("change", function () {
    console.log("Dropdown 2: ", dropdown2.value);
});