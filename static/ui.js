// Get the sliders
const positionSlider = document.getElementById("position_slider");
const ChannelSlider = document.getElementById("channel_slider");
// Get the dropdown menus
const dropdown1 = document.getElementById("dropdown1");
const dropdown2 = document.getElementById("dropdown2");
// Add event listeners
positionSlider.addEventListener("input", function () {
    console.log("Position: ", positionSlider.value);
});
ChannelSlider.addEventListener("input", function () {
    console.log("Channel: ", ChannelSlider.value);
});
dropdown1.addEventListener("change", function () {
    console.log("Dropdown 1: ", dropdown1.value);
});
dropdown2.addEventListener("change", function () {
    console.log("Dropdown 2: ", dropdown2.value);
});