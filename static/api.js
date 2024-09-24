positionSlider.addEventListener("input", updateImage, updatePositionText);
channelSlider.addEventListener("input", updateImage, updateChannelText);
timeframeSlider.addEventListener("input", updateImage, updateTimeframeText);
particleSlider.addEventListener("input", updateImage, updateParticleText);

// window.updateImage = function () {
function updateImage() {
  const params = {
    position: document.getElementById("position_slider").value,
    channel: document.getElementById("channel_slider").value,
    frame: document.getElementById("timeframe_slider").value,
    particle: document.getElementById("particle_slider").value,
    // contrast: contrast.noUiSlider.get(),
    // overlay: document.getElementById('overlay').checked,
    // overlay_contrast: overlay_slider.noUiSlider.get()
    //        brightness: brightnessSlider.noUiSlider.get(),  // Example for adding a new slider
    //        saturation: document.getElementById('saturation-slider').value  // Example for another type of input
  };

  fetchImageUpdate("/update_image", params);
}

function updatePositionText() {
  document.getElementById("position_value").textContent =
    `${positionSlider.value}/${positionSlider.max}`;
}
function updateChannelText() {
  document.getElementById("channel_value").textContent =
    `${channelSlider.value}/${channelSlider.max}`;
}

function updateTimeframeText() {
  document.getElementById("timeframe_value").textContent =
    `${timeframeSlider.value}/${timeframeSlider.max}`;
}

function updateParticleText() {
  document.getElementById("particle_value").textContent =
    `${particleSlider.value}/${particleSlider.max}`;
}

function createRequestBody(params) {
  let body = Object.keys(params)
    .map(
      (key) => `${encodeURIComponent(key)}=${encodeURIComponent(params[key])}`,
    )
    .join("&");
  // Replace encoded commas if necessary
  body = body.replace(/%2C/g, ",");
  return body;
}

function fetchImageUpdate(url, params) {
  const requestBody = createRequestBody(params); // channel=0&contrast=22112%2C65535
  fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body: requestBody,
  })
    .then((response) => response.json())
    .then((data) => updateImageDisplay(data.channel_image));
}

function updateImageDisplay(base64Image) {
  const imgSrc = `data:image/jpeg;base64,${base64Image}`;
  document.getElementById("channel_image").src = imgSrc;
}