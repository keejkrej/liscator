// Get the sliders
const positionSlider = document.getElementById("position_slider");
const ChannelSlider = document.getElementById("channel_slider");

document.addEventListener('DOMContentLoaded', (event) => {
    let channel = document.getElementById("channel_slider");
    let channelValue = document.getElementById("channel_value");
    
        channel.addEventListener('input', function () {
        channelValue.innerHTML = channel.value;
    });
});

document.addEventListener('DOMContentLoaded', (event) => {
    let position = document.getElementById("position_slider");
    let positionValue = document.getElementById("position_value");

    position.addEventListener('input', function () {
        positionValue.innerHTML = position.value;
    });
});

document.addEventListener('DOMContentLoaded', (event) => {
    let timeframe = document.getElementById("timeframe_slider");
    let timeframeValue = document.getElementById("timeframe_value");

    timeframe.addEventListener('input', function () {
        timeframeValue.innerHTML = timeframe.value;
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