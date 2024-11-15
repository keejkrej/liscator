/**
 * api.js
 * Handles all API interactions and updates to the image and plot displays
 * based on user input from sliders and other controls.
 */

// Set up event listeners for slider changes
positionSlider.addEventListener("input", updateImageAndPlot);
channelSlider.addEventListener("input", updateImage);
timeframeSlider.addEventListener("input", updateImage);
particleSlider.addEventListener("input", updateImageAndPlot);

/**
 * Updates both the image and brightness plot
 */
function updateImageAndPlot() {
  updateImage();
  updateBrightnessPlot();
}

/**
 * Fetches and updates the image based on current slider values
 */
function updateImage() {
  const params = {
    position: document.getElementById("position_slider").value,
    channel: document.getElementById("channel_slider").value,
    frame: document.getElementById("timeframe_slider").value,
    particle: document.getElementById("particle_slider").value,
  };

  fetchImageUpdate("/update_image", params);
}

/**
 * Updates the brightness plot based on position and particle values
 */
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

/**
 * Helper functions to update slider value displays
 */
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

/**
 * Creates URL-encoded request body from parameters object
 * @param {Object} params - Parameters to encode
 * @returns {string} Encoded parameter string
 */
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

/**
 * Fetches image updates from the server and updates the display
 * @param {string} url - API endpoint
 * @param {Object} params - Request parameters
 */
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
      if (data.all_particles_len !== undefined) {
        const particleSlider = document.getElementById("particle_slider");
        particleSlider.max = data.all_particles_len - 1;
        document.getElementById("particle_value").innerHTML =
          `${particleSlider.value}/${data.all_particles_len - 1}`;
      }
      if (data.brightness_plot) {
        Plotly.react(
          "brightness-plot",
          JSON.parse(data.brightness_plot).data,
          JSON.parse(data.brightness_plot).layout,
        );
      }
    });
}

/**
 * Updates the image element with new base64 encoded image data
 * @param {string} base64Image - Base64 encoded image data
 */
function updateImageDisplay(base64Image) {
  const imgSrc = `data:image/jpeg;base64,${base64Image}`;
  document.getElementById("channel_image").src = imgSrc;
}
