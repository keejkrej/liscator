positionSlider.addEventListener("input", updateImageAndPlot);
channelSlider.addEventListener("input", updateImage);
timeframeSlider.addEventListener("input", updateImage);
particleSlider.addEventListener("input", updateImageAndPlot);

function updateImageAndPlot() {
  updateImage();
  updateBrightnessPlot();
}

function updateImage() {
  const params = {
    position: document.getElementById("position_slider").value,
    channel: document.getElementById("channel_slider").value,
    frame: document.getElementById("timeframe_slider").value,
    particle: document.getElementById("particle_slider").value,
  };

  fetchImageUpdate("/update_image", params);
}

function updateBrightnessPlot() {
  const params = {
    position: document.getElementById("position_slider").value,
    particle: document.getElementById("particle_slider").value,
  };

  fetch("/update_image", {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body: createRequestBody(params),
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.brightness_plot) {
        Plotly.react(
          "brightness-plot",
          JSON.parse(data.brightness_plot).data,
          JSON.parse(data.brightness_plot).layout,
        );
      }
    });
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
  const requestBody = createRequestBody(params);
  fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body: requestBody,
  })
    .then((response) => response.json())
    .then((data) => {
      updateImageDisplay(data.channel_image);
      if (data.brightness_plot) {
        Plotly.react(
          "brightness-plot",
          JSON.parse(data.brightness_plot).data,
          JSON.parse(data.brightness_plot).layout,
        );
      }
    });
}

function updateImageDisplay(base64Image) {
  const imgSrc = `data:image/jpeg;base64,${base64Image}`;
  document.getElementById("channel_image").src = imgSrc;
}
