/**
 * keyboard_shortcuts.js
 * Handles keyboard shortcuts for controlling sliders
 */

/**
 * Updates slider value and its text display
 * @param {HTMLElement} slider - Slider element to update
 * @param {number} value - New value for the slider
 */
function updateSliderAndText(slider, value) {
  slider.value = value;
  const textElement = document.getElementById(slider.id + "_value");
  if (textElement) {
    textElement.textContent = value + "/" + slider.max;
  }
  // Trigger the input event
  slider.dispatchEvent(new Event("input"));
}

// Add keyboard event listener for slider controls
// Shift + Arrow keys: Position and Channel
// Alt + Arrow keys: Timeframe and Particle
// Space: Toggle particle enabled/disabled
document.addEventListener("keydown", (event) => {
  let updateNeeded = false;

  if (event.code === "Space") {
    event.preventDefault();
    const checkbox = document.getElementById("particle_enabled");
    if (checkbox) {
      checkbox.checked = !checkbox.checked;
      checkbox.dispatchEvent(new Event("change"));
      updateNeeded = true;
    }
  } else if (event.shiftKey) {
    switch (event.key) {
      case "ArrowUp":
        event.preventDefault();
        updateSliderAndText(positionSlider, parseInt(positionSlider.value) + 1);
        updateNeeded = true;
        break;
      case "ArrowDown":
        event.preventDefault();
        updateSliderAndText(positionSlider, parseInt(positionSlider.value) - 1);
        updateNeeded = true;
        break;
      case "ArrowRight":
        event.preventDefault();
        updateSliderAndText(channelSlider, parseInt(channelSlider.value) + 1);
        updateNeeded = true;
        break;
      case "ArrowLeft":
        event.preventDefault();
        updateSliderAndText(channelSlider, parseInt(channelSlider.value) - 1);
        updateNeeded = true;
        break;
    }
  } else if (event.altKey) {
    switch (event.key) {
      case "ArrowUp":
        event.preventDefault();
        updateSliderAndText(
          timeframeSlider,
          parseInt(timeframeSlider.value) + 1,
        );
        updateNeeded = true;
        break;
      case "ArrowDown":
        event.preventDefault();
        updateSliderAndText(
          timeframeSlider,
          parseInt(timeframeSlider.value) - 1,
        );
        updateNeeded = true;
        break;
      case "ArrowRight":
        event.preventDefault();
        updateSliderAndText(particleSlider, parseInt(particleSlider.value) + 1);
        updateNeeded = true;
        break;
      case "ArrowLeft":
        event.preventDefault();
        updateSliderAndText(particleSlider, parseInt(particleSlider.value) - 1);
        updateNeeded = true;
        break;
    }
  }

  if (updateNeeded) {
    updateImage();
  }
});
