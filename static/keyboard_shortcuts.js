// Sliders from ui.js
// const positionSlider = document.getElementById("position_slider");
// const channelSlider = document.getElementById("channel_slider");
// const timeframeSlider = document.getElementById("timeframe_slider");
// const particleSlider = document.getElementById("particle_slider");

// Event listeners for sliders
document.addEventListener("keydown", (event) => {
  if (event.shiftKey) {
    switch (event.key) {
      case "ArrowUp":
        positionSlider.value = parseInt(positionSlider.value) + 1;
        break;
      case "ArrowDown":
        positionSlider.value = parseInt(positionSlider.value) - 1;
        break;
      case "ArrowRight":
        channelSlider.value = parseInt(channelSlider.value) + 1;
        break;
      case "ArrowLeft":
        channelSlider.value = parseInt(channelSlider.value) - 1;
        break;
    }
  } else if (event.ctrlKey) {
    switch (event.key) {
      case "ArrowUp":
        timeframeSlider.value = parseInt(timeframeSlider.value) + 1;
        break;
      case "ArrowDown":
        timeframeSlider.value = parseInt(timeframeSlider.value) - 1;
        break;
      case "ArrowRight":
        particleSlider.value = parseInt(particleSlider.value) + 1;
        break;
      case "ArrowLeft":
        particleSlider.value = parseInt(particleSlider.value) - 1;
        break;
    }
  }
});
