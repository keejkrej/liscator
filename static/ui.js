// Get the sliders
const positionSlider = document.getElementById("position_slider");
const channelSlider = document.getElementById("channel_slider");
const timeframeSlider = document.getElementById("timeframe_slider");
const particleSlider = document.getElementById("particle_slider");

document.addEventListener("DOMContentLoaded", (event) => {
  let channel = document.getElementById("channel_slider");
  let channelValue = document.getElementById("channel_value");
  let n_channels = channel.max;

  channel.addEventListener("input", function () {
    channelValue.innerHTML = `${channel.value}/${n_channels}`;
  });
});

document.addEventListener("DOMContentLoaded", (event) => {
  let position = document.getElementById("position_slider");
  let positionValue = document.getElementById("position_value");
  let n_positions = parseInt(position.max) + 1;

  position.addEventListener("input", function () {
    positionValue.innerHTML = `${position.value}/${n_positions - 1}`;
  });
});

document.addEventListener("DOMContentLoaded", (event) => {
  let timeframe = document.getElementById("timeframe_slider");
  let timeframeValue = document.getElementById("timeframe_value");
  let n_frames = timeframe.max;

  timeframe.addEventListener("input", function () {
    timeframeValue.innerHTML = `${timeframe.value}/${n_frames}`;
  });
});

document.addEventListener("DOMContentLoaded", (event) => {
  let particle = document.getElementById("particle_slider");
  let particleValue = document.getElementById("particle_value");
  let all_particles_len = parseInt(particle.max) + 1;

  particle.addEventListener("input", function () {
    particleValue.innerHTML = `${particle.value}/${all_particles_len - 1}`;
  });
});

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
