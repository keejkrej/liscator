function updateSliderAndText(slider, value) {
  slider.value = value;
  const textElement = document.getElementById(slider.id + "_value");
  if (textElement) {
    textElement.textContent = value + "/" + slider.max;
  }
  // Trigger the input event
  slider.dispatchEvent(new Event("input"));
}

document.addEventListener("keydown", (event) => {
  let updateNeeded = false;

  if (event.shiftKey) {
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
        updateSliderAndText(timeframeSlider, parseInt(timeframeSlider.value) + 1);
        updateNeeded = true;
        break;
      case "ArrowDown":
        event.preventDefault();
        updateSliderAndText(timeframeSlider, parseInt(timeframeSlider.value) - 1,);
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