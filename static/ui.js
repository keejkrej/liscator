// Get the sliders
const positionSlider = document.getElementById("position_slider");
const channelSlider = document.getElementById("channel_slider");
const timeframeSlider = document.getElementById("timeframe_slider");
const particleSlider = document.getElementById("particle_slider");

document.addEventListener("DOMContentLoaded", (event) => {
  let sliders = ["position", "channel", "timeframe", "particle"];
  sliders.forEach((slider) => {
    let sliderElement = document.getElementById(`${slider}_slider`);
    let sliderValue = document.getElementById(`${slider}_value`);
    let n_values = parseInt(sliderElement.max) + 1;

    sliderElement.addEventListener("input", function () {
      sliderValue.innerHTML = `${sliderElement.value}/${n_values - 1}`;
    });
  });
});

// Get the dropdown menus
const dropdown1 = document.getElementById("dropdown1");
const dropdown2 = document.getElementById("dropdown2");
// Add event listeners
positionSlider.addEventListener("input", function () {
  console.log("Position: ", positionSlider.value);
});
channelSlider.addEventListener("input", function () {
  console.log("Channel: ", channelSlider.value);
});
dropdown1.addEventListener("change", function () {
  console.log("Dropdown 1: ", dropdown1.value);
});
dropdown2.addEventListener("change", function () {
  console.log("Dropdown 2: ", dropdown2.value);
});
