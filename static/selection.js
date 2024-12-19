/**
 * selection.js
 * Implements file browser functionality for selecting input and output directories
 */

/**
 * Creates a file browser instance
 * @param {string} currentPathId - ID of current path element
 * @param {string} upDirId - ID of up directory button
 * @param {string} selectFolderId - ID of select folder button
 * @param {string} fileListId - ID of file list container
 * @param {string} selectedItemId - ID of selected item display
 * @returns {Object} File browser controller object
 */
function createFileBrowser(
  currentPathId,
  upDirId,
  selectFolderId,
  fileListId,
  selectedItemId,
) {
  const currentPath = document.getElementById(currentPathId);
  const upDir = document.getElementById(upDirId);
  const selectFolder = document.getElementById(selectFolderId);
  const fileList = document.getElementById(fileListId);
  const selectedItem = document.getElementById(selectedItemId);

  /**
   * Loads and displays the contents of a directory
   * @param {string} path - Path to load
   */
  async function loadDirectory(path) {
    try {
      const response = await fetch(
        `/list_directory?path=${encodeURIComponent(path)}`,
      );
      const data = await response.json();

      if (data.error) {
        console.error(data.error);
        return;
      }

      currentPath.value = data.path;
      fileList.innerHTML = "";

      // Sort items alphabetically
      data.items.sort((a, b) => {
        // First, sort directories before files
        if (a.isDirectory && !b.isDirectory) return -1;
        if (!a.isDirectory && b.isDirectory) return 1;
        // Then sort alphabetically by name
        return a.name.localeCompare(b.name);
      });

      data.items.forEach((item) => {
        const div = document.createElement("div");
        div.className =
          "p-2 hover:bg-gray-100 cursor-pointer flex items-center";

        const icon = document.createElement("span");
        icon.className = "mr-2";
        icon.textContent = item.isDirectory ? "📁" : "📄";
        div.appendChild(icon);

        const name = document.createElement("span");
        name.textContent = item.name;
        div.appendChild(name);

        div.onclick = () => {
          const newPath =
            data.path + (data.path.endsWith("/") ? "" : "/") + item.name;
          if (item.isDirectory) {
            loadDirectory(newPath);
          } else {
            selectedItem.textContent = `Selected File: ${newPath}`;
          }
        };
        fileList.appendChild(div);
      });
    } catch (error) {
      console.error("Error loading directory:", error);
    }
  }

  // Handle navigation to parent directory
  upDir.onclick = () => {
    const parentPath =
      currentPath.value.split("/").slice(0, -1).join("/") || "/";
    loadDirectory(parentPath);
  };

  // Handle folder selection
  selectFolder.onclick = () => {
    selectedItem.textContent = `Selected Folder: ${currentPath.value}`;
  };

  // Initial load
  loadDirectory("/");

  return { loadDirectory };
}

// Create two file browsers
const nd2Browser = createFileBrowser(
  "nd2CurrentPath",
  "nd2UpDir",
  "nd2SelectFolder",
  "nd2FileList",
  "nd2SelectedItem",
);
const outBrowser = createFileBrowser(
  "outCurrentPath",
  "outUpDir",
  "outSelectFolder",
  "outFileList",
  "outSelectedItem",
);

/**
 * Submits selected paths to the server
 * @param {string} destination - Redirect destination after submission
 */
async function submitPaths(destination) {
  const nd2Path =
    document.getElementById("nd2SelectedItem").textContent.split(": ")[1] ||
    "/";
  const outPath =
    document.getElementById("outSelectedItem").textContent.split(": ")[1] ||
    "/";

  try {
    const response = await fetch("/select_paths", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        nd2_path: nd2Path,
        out_path: outPath,
        redirect_to: destination,
      }),
    });
    const data = await response.json();
    if (data.error) {
      alert(data.error);
    } else if (data.redirect) {
      window.location.href = data.redirect;
    }
  } catch (error) {
    console.error("Error submitting paths:", error);
    alert("An error occurred. Please try again.");
  }
}
