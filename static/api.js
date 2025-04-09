/**
 * api.js
 * Handles all API interactions and updates to the image and plot displays
 * based on user input from sliders and other controls.
 */

let updateInProgress = false;
let pendingUpdate = false;
let debounceTimeout = null;
const DEBOUNCE_DELAY = 300;

// Set up event listeners for slider changes
positionSlider.addEventListener("input", debouncedUpdateImageAndPlot);
channelSlider.addEventListener("input", updateImage);
timeframeSlider.addEventListener("input", updateImage);
particleSlider.addEventListener("input", debouncedUpdateImageAndPlot);
const particleEnabledCheckbox = document.getElementById("particle_enabled");

particleEnabledCheckbox.addEventListener("change", function () {
  updateParticleEnabled(this.checked);
});

function debouncedUpdateImageAndPlot() {
  clearTimeout(debounceTimeout);
  debounceTimeout = setTimeout(() => {
    if (!updateInProgress) {
      updateImageAndPlot();
    } else {
      pendingUpdate = true;
    }
  }, DEBOUNCE_DELAY);
}

/**
 * Updates both the image and brightness plot
 */
function updateImageAndPlot() {
  if (updateInProgress) {
    pendingUpdate = true;
    return;
  }

  updateInProgress = true;
  showLoadingIndicator();

  const params = {
    position: document.getElementById("position_slider").value,
    channel: document.getElementById("channel_slider").value,
    frame: document.getElementById("timeframe_slider").value,
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
      updateImageDisplay(data.channel_image);
      if (data.brightness_plot) {
        Plotly.react(
          "brightness-plot",
          JSON.parse(data.brightness_plot).data,
          JSON.parse(data.brightness_plot).layout,
        );
      }
      if (data.all_particles_len !== undefined) {
        const particleSlider = document.getElementById("particle_slider");
        particleSlider.max = data.all_particles_len;
      }
      if (data.disabled_particles !== undefined) {
        particleEnabledCheckbox.checked = !data.disabled_particles.includes(
          parseInt(params.particle),
        );
      }
    })
    .catch((error) => console.error("Error:", error))
    .finally(() => {
      updateInProgress = false;
      hideLoadingIndicator();
      if (pendingUpdate) {
        pendingUpdate = false;
        updateImageAndPlot();
      }
    });
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

function showLoadingIndicator() {
  document.body.style.cursor = "wait";
}

function hideLoadingIndicator() {
  document.body.style.cursor = "default";
}

function updateParticleEnabled(enabled) {
  fetch("/update_particle_enabled", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      enabled: enabled,
    }),
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.channel_image) {
        updateImageDisplay(data.channel_image);
      }
      if (data.brightness_plot) {
        Plotly.react(
          "brightness-plot",
          JSON.parse(data.brightness_plot).data,
          JSON.parse(data.brightness_plot).layout,
        );
      }
    })
    .catch((error) => console.error("Error:", error));
}

/* *
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

/* *
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

/* *
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
        particleSlider.max = data.all_particles_len;
      }
      if (data.brightness_plot) {
        Plotly.react(
          "brightness-plot",
          JSON.parse(data.brightness_plot).data,
          JSON.parse(data.brightness_plot).layout,
        );
      }
      // Update checkbox state based on disabled particles
      //
      if (data.disabled_particles !== undefined) {
        particleEnabledCheckbox.checked = !data.disabled_particles.includes(
          parseInt(params.particle), // Convert to integer since it might be a string
        );
      }

      // if (
      //   data.current_particle !== undefined &&
      //   data.disabled_particles !== undefined
      // ) {
      //   particleEnabledCheckbox.checked = !data.disabled_particles.includes(
      //     data.current_particle - 1,
      //   );
      // }
    });
}

/* *
 * Updates the image element with new base64 encoded image data
 * @param {string} base64Image - Base64 encoded image data
 */
function updateImageDisplay(base64Image) {
  const imgSrc = `data:image/jpeg;base64,${base64Image}`;
  document.getElementById("channel_image").src = imgSrc;
}
