import axios from "axios";
import * as am4core from "@amcharts/amcharts4/core";
import * as am4charts from "@amcharts/amcharts4/charts";
import am4themes_animated from "@amcharts/amcharts4/themes/animated";
import moment from "moment";

declare global {
  interface Window {
    io: Function;
  }
}

const apiUrl = process.env.NODE_ENV === 'development' ? "http://localhost:5000" : "REPLACE_WITH_PROD_URL";

function init() {

  // Connect to the Socket.IO server.
  // The connection URL has the following format, relative to the current page:
  //     http[s]://<domain>:<port>[/<namespace>]
  const socket = window.io();

  // Event handler for new connections.
  // The callback function is invoked when a connection with the
  // server is established.
  // socket.on("connect", function () {
  //   socket.emit("my_event", { data: "I'm connected!" });
  // });

  am4core.useTheme(am4themes_animated);

  // Create chart instance
  const chart = am4core.create("chartdiv", am4charts.XYChart);

  type BpmDataPoint = {
    date: Date | null;
    value: number | null;
    label: string;
  };

  const userData = {
    ageRange: null as null | AgeRange,
    currentBPM: null as null | number,
    historyBPM: [] as BpmDataPoint[],
    currentEmotion: null,
    gender: null,
  };

  async function initProgramSequence() {
    /**
     * The basic logic is that all ML processing models require
     * a suitable grab cut of the face; without it, sequential
     * requests will also fail; therefore, we should not loop and
     * look for further data until the initial model returns
     */
    if (!userData.gender) await genderXHR();

    if (userData.gender && !userData.ageRange) await ageXHR();

    if (userData.gender && userData.ageRange) await emotionXHR(); //TODO: Move to socket or event listener

    if (userData.gender && userData.ageRange && userData.currentEmotion)
      await getHeartbeat(); //TODO: Move to socket or event listener

    setTimeout(initProgramSequence, 3000);
  }

  async function genderXHR() {
    await axios
      .get("/get-gender")
      .then(function (res) {
        userData.gender = res.data.Gender[0];
        var metaDataGender = document.getElementById("pulse-meta_data_gender");
        if (!!metaDataGender)
          metaDataGender.innerHTML = "Gender: " + userData.gender;
      })
      .catch((err) => {
        console.error(err);
      });
  }

  async function ageXHR() {
    await axios
      .get("/get-age")
      .then(function (res) {
        console.log(res);
        const age = res.data.Age;
        userData.ageRange = ageRangeFit(age);
        var metaDataAge = document.getElementById("pulse-meta_data_age");
        if (!!metaDataAge) metaDataAge.innerHTML = "Age: " + userData.ageRange;
      })
      .catch((err) => {
        console.error(err);
      });
  }

  async function emotionXHR() {
    await axios
      .get("/get-emotion")
      .then(function (res) {
        console.log(res);
        const emotion = res.data.Emotion[0];
        if (!!emotion && emotion.length >= 0) {
          userData.currentEmotion = emotion;
          var metaDataEmotion = document.getElementById(
            "pulse-meta_data_emotion"
          );
          if (!!metaDataEmotion)
            metaDataEmotion.innerHTML = "Emotion: " + userData.currentEmotion;
        }
      })
      .catch((err) => {
        userData.currentEmotion = null;
        console.error(err);
      });
  }

  function ageRangeFit(age: number) {
    if (age < 7) return "0-7";
    else if (age < 14) return "7-14";
    else if (age < 21) return "14 - 21";
    else if (age < 28) return "21 - 28";
    else if (age < 36) return "28 - 36";
    else if (age < 42) return "36 - 42";
    else if (age < 50) return "42 - 50";
    else if (age < 60) return "50.5 - 60";
    else if (age < 100) return "60 - 100";
    else return "Out of range";
  }

  type AgeRange = ReturnType<typeof ageRangeFit>;

  async function getHeartbeat() {
    // make Ajax call here, inside the callback call:
    // ...
    await axios
      .get("/heartbeat")
      .then(function (response) {
        // handle success
        var div = document.getElementById("pulse-bpm_count");

        if (!!div && Array.isArray(response.data.bpm)) {
          userData.currentBPM = Math.round(response.data.bpm[0]);
          div.innerHTML = userData.currentBPM + "<small>BPM</small>";
        }

        var bpmBox = document.getElementById("p-bpm_text");
        if (!!bpmBox) bpmBox.style.display = "flex";

        const bpmData: BpmDataPoint = {
          date: moment().toDate(),
          value: userData.currentBPM,
          label: moment().format("HH:mm"),
        };

        if (bpmData.value != 0) userData.historyBPM.push(bpmData);

        var chartUI = document.getElementById("chartdiv");
        if (userData.historyBPM.length > 1 && chartUI) {
          chartUI.style.display = "flex";
        }

        chart.data = userData.historyBPM;
        console.log(userData.historyBPM);
      })
      .catch(function (error) {
        // handle error
        console.log(error);
      });
  }

  /**
   * --------------------------------------- *
   * For more information visit:
   * https://www.amcharts.com/
   *
   * Documentation is available at:
   * https://www.amcharts.com/docs/v4/
   * ---------------------------------------
   */
  function drawChart() {
    chart.data = [];

    // Create axes
    var dateAxis = chart.xAxes.push(new am4charts.DateAxis());
    dateAxis.renderer.grid.template.location = 0.5;
    dateAxis.renderer.minGridDistance = 50;

    dateAxis.baseInterval = {
      timeUnit: "second",
      count: 1,
    };

    var valueAxis = chart.yAxes.push(new am4charts.ValueAxis());

    // Create series
    function createSeries(field: string, name: string) {
      var series = chart.series.push(new am4charts.LineSeries());
      series.dataFields.valueY = field;
      series.dataFields.dateX = "date";
      series.name = name;
      series.tooltipText = "{label}: [b]{valueY}[/]bpm";
      if (series.tooltip) {
        series.tooltip.getFillFromObject = false;
        series.tooltip.background.fill = am4core.color("red");
      }
      series.strokeWidth = 2;
      series.stroke = am4core.color("red");

      var bullet = series.bullets.push(new am4charts.CircleBullet());
      bullet.circle.fill = am4core.color("red");
      bullet.circle.stroke = am4core.color("#fff");
      bullet.circle.strokeWidth = 2;
    }

    createSeries("value", "Series #1");

    chart.cursor = new am4charts.XYCursor();

    return chart;
  }
  // initial call, or just call refresh directly
  setTimeout(initProgramSequence, 5000);
  drawChart();

  // getUserMedia JS
  if (!navigator.mediaDevices.getUserMedia) {
    throw Error("Your browser does not support this application.");
  }

  const dimensions = {
    width: 500, 
    height: 375,
  }

  const video = document.querySelector<HTMLVideoElement>("#videoElement")!;

  const canvas = document.createElement("canvas");
  canvas.style.opacity = "0";
  canvas.style.position = "fixed";
  canvas.style.transform = 'translate(-100%,-100%)';
  canvas.width = dimensions.width;
  canvas.height = dimensions.height;
  const context = canvas.getContext("2d")!;
  const track = (canvas as any).captureStream().getVideoTracks()[0];

  document.body.append(canvas);

  const frames: { id: number; blob: Blob | null }[] = [];

  function renderVideoToCanvas() {
    animate();
    return track;
    function animate() {
      context.drawImage(video, 0, 0, dimensions.width, dimensions.height);
      if (track.readyState === "live") {
        requestAnimationFrame(animate);
      }
    }
  }

  let nextFrameId = 0;
  function saveFrame(blob: Blob | null) {
    if(!blob) return

    frames.push({ id: nextFrameId++, blob });
    const srcBlob = URL.createObjectURL(blob);

    if (!srcBlob) return;
    const file = new File([blob], "Frame.jpeg", { type: "image/jpeg" });

    socket.emit("handle_frame", file);
    console.log("frame sent");
  }

  function startCapturingFrames() {
    const handler = function () {
      canvas.toBlob(saveFrame, "image/jpeg", 0.95);
      // renderPreviewsForCapturedFrames();
    };
    setInterval(handler, .05 * 1000);
    handler();
  }

  function renderPreviewsForCapturedFrames() {
    frames.map((b) => {
      const id = `captured-frame-${b.id}`;
      if (document.getElementById(id)) return;
      const img = document.createElement("img");
      img.classList.add('captured-frame-preview-image')
      img.id = id;
      img.src = URL.createObjectURL(b.blob);
      img.width = dimensions.width / 5;
      img.height = dimensions.height / 5;
      document.body.append(img);
    });
  }

  function watchForWindowResize() {
    window.addEventListener('resize', () => {
      const videoStyle = getComputedStyle(video);
      const { width, height } = videoStyle;
      dimensions.width = +width || 500;
      dimensions.height = +height || 375;
      canvas.width = dimensions.width;
      canvas.height = dimensions.width;
      document.querySelectorAll<HTMLImageElement>(".captured-frame-preview-image").forEach(image => {
        image.width = dimensions.width / 5;
        image.height = dimensions.height / 5;
      });
    });
  }

  navigator.mediaDevices
    .getUserMedia({
      video: true,
    })
    .then(function (stream) {
      video.srcObject = stream;
      renderVideoToCanvas();
      startCapturingFrames();
      watchForWindowResize();
    })
    .catch(function (error) {
      console.error(error);
    });

  function arrayBufferToBase64(buffer: ArrayBuffer) {
    var binary = "";
    var bytes = new Uint8Array(buffer);
    var len = bytes.byteLength;
    for (var i = 0; i < len; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return window.btoa(binary);
  }

  socket.on("image_processed", function (imageMediaBlob: { media: Blob }) {
    console.log("appending media...");
    const imageEl = document.getElementById(
      "processedImage"
    ) as HTMLImageElement | null;
  });
}

document.addEventListener("DOMContentLoaded", init);