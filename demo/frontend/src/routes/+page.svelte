<script lang="ts">
  import { onMount } from 'svelte';
  import type { Fields, PipelineInfo } from '$lib/types';
  import { PipelineMode } from '$lib/types';
  import ImagePlayer from '$lib/components/ImagePlayer.svelte';
  import VideoInput from '$lib/components/VideoInput.svelte';
  import Button from '$lib/components/Button.svelte';
  import PipelineOptions from '$lib/components/PipelineOptions.svelte';
  import Spinner from '$lib/icons/spinner.svelte';
  import Warning from '$lib/components/Warning.svelte';
  import { lcmLiveStatus, lcmLiveActions, LCMLiveStatus, streamId } from '$lib/lcmLive';
  import { mediaStreamActions, onFrameChangeStore } from '$lib/mediaStream';
  import { getPipelineValues, deboucedPipelineValues } from '$lib/store';

  let pipelineParams: Fields;
  let pipelineInfo: PipelineInfo;
  let pageContent: string;
  let isImageMode: boolean = false;
  let maxQueueSize: number = 0;
  let currentQueueSize: number = 0;
  let queueCheckerRunning: boolean = false;
  let warningMessage: string = '';
  let inputMode: 'camera' | 'upload' = 'camera';
  let uploadedFile: File | null = null;
  let uploadedVideoUrl: string | null = null;
  let fileInputEl: HTMLInputElement | null = null;
  onMount(() => {
    getSettings();
  });

  async function getSettings() {
    const settings = await fetch('/api/settings').then((r) => r.json());
    pipelineParams = settings.input_params.properties;
    pipelineInfo = settings.info.properties;
    isImageMode = pipelineInfo.input_mode.default === PipelineMode.IMAGE;
    maxQueueSize = settings.max_queue_size;
    pageContent = settings.page_content;
    console.log(pipelineParams);
    toggleQueueChecker(true);
  }
  function toggleQueueChecker(start: boolean) {
    queueCheckerRunning = start && maxQueueSize > 0;
    if (start) {
      getQueueSize();
    }
  }
  async function getQueueSize() {
    if (!queueCheckerRunning) {
      return;
    }
    const data = await fetch('/api/queue').then((r) => r.json());
    currentQueueSize = data.queue_size;
    setTimeout(getQueueSize, 10000);
  }

  function getStreamData() {
    if (isImageMode) {
      return [getPipelineValues(), $onFrameChangeStore?.blob];
    } else {
      return [$deboucedPipelineValues];
    }
  }

  $: isLCMRunning = $lcmLiveStatus !== LCMLiveStatus.DISCONNECTED;
  $: if ($lcmLiveStatus === LCMLiveStatus.TIMEOUT) {
    warningMessage = 'Session timed out. Please try again.';
  }
  let disabled = false;
  let isStreaming = false;

  function handleVideoEnded() {
    isStreaming = false;
    lcmLiveActions.stop();
  }

  function handleInputModeChange(mode: 'camera' | 'upload') {
    // Always stop LCM and media streams first
    lcmLiveActions.stop();
    mediaStreamActions.stop();
    isStreaming = false;
    disabled = false;

    // Reset stores and file input
    onFrameChangeStore.set({ blob: new Blob() });
    streamId.set(null); // <-- force output reset
    lcmLiveStatus.set(LCMLiveStatus.DISCONNECTED); // <-- force status reset
    if (uploadedVideoUrl) {
      URL.revokeObjectURL(uploadedVideoUrl);
      uploadedVideoUrl = null;
    }
    uploadedFile = null;
    if (fileInputEl) fileInputEl.value = '';

    // Set new mode
    inputMode = mode;
  }

  function handleFileChange(event: Event) {
    const files = (event.target as HTMLInputElement).files;
    if (files && files.length > 0) {
      uploadedFile = files[0];
      if (uploadedVideoUrl) URL.revokeObjectURL(uploadedVideoUrl);
      uploadedVideoUrl = URL.createObjectURL(uploadedFile);
    }
  }

  async function toggleLcmLive() {
    try {
      if (!isLCMRunning) {
        if (inputMode === 'camera') {
          await mediaStreamActions.enumerateDevices();
          await mediaStreamActions.start();
        }
        disabled = true;
        isStreaming = true;
        await lcmLiveActions.start(getStreamData);
        disabled = false;
        toggleQueueChecker(false);
      } else {
        if (inputMode === 'camera') {
          mediaStreamActions.stop();
        }
        lcmLiveActions.stop();
        isStreaming = false;
        streamId.set(null); // <-- force output reset
        lcmLiveStatus.set(LCMLiveStatus.DISCONNECTED); // <-- force status reset
        toggleQueueChecker(true);
      }
    } catch (e) {
      warningMessage = e instanceof Error ? e.message : '';
      disabled = false;
      isStreaming = false;
      if (inputMode === 'camera') {
        mediaStreamActions.stop();
      }
      lcmLiveActions.stop();
      streamId.set(null); // <-- force output reset
      lcmLiveStatus.set(LCMLiveStatus.DISCONNECTED); // <-- force status reset
      toggleQueueChecker(true);
    }
  }
</script>

<svelte:head>
  <script
    src="https://cdnjs.cloudflare.com/ajax/libs/iframe-resizer/4.3.9/iframeResizer.contentWindow.min.js"
  ></script>
</svelte:head>

<main class="container mx-auto flex max-w-5xl flex-col gap-3 px-4 py-4">
  <Warning bind:message={warningMessage}></Warning>
  <article class="text-center">
    {#if pageContent}
      {@html pageContent}
    {/if}
    {#if maxQueueSize > 0}
      <p class="text-sm">
        There are <span id="queue_size" class="font-bold">{currentQueueSize}</span>
        user(s) sharing the same GPU, affecting real-time performance. Maximum queue size is {maxQueueSize}.
        <a
          href="https://huggingface.co/spaces/radames/Real-Time-Latent-Consistency-Model?duplicate=true"
          target="_blank"
          class="text-blue-500 underline hover:no-underline">Duplicate</a
        > and run it on your own GPU.
      </p>
    {/if}
    <p class="text-sm mt-2 text-gray-600">You can use your camera or upload a video as input. The left panel will show input frames, and the right panel will show streaming output frames.</p>
  </article>
  {#if pipelineParams}
    <!-- Input mode controls: now above the grid, spanning the whole page -->
    <div class="mb-4 flex flex-row items-center justify-center gap-4">
      <label>
        <input type="radio" name="inputMode" value="camera" bind:group={inputMode} on:change={() => handleInputModeChange('camera')} disabled={isStreaming} />
        Camera
      </label>
      <label>
        <input type="radio" name="inputMode" value="upload" bind:group={inputMode} on:change={() => handleInputModeChange('upload')} disabled={isStreaming} />
        Upload Video
      </label>
      {#if inputMode === 'upload'}
        <input type="file" accept="video/*" on:change={handleFileChange} disabled={isStreaming} class="ml-2" bind:this={fileInputEl} />
      {/if}
    </div>
    <article class="my-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
      {#if isImageMode}
        <div class="sm:col-start-1">
          <VideoInput
            width={Number(pipelineParams.width.default)}
            height={Number(pipelineParams.height.default)}
            isStreaming={isStreaming}
            onVideoEnded={handleVideoEnded}
            onInputModeChange={() => handleInputModeChange(inputMode)}
            bind:inputModeProp={inputMode}
            { ...(inputMode === 'upload' ? { uploadedVideoUrl } : {}) }
          />
        </div>
      {/if}
      <div class={isImageMode ? 'sm:col-start-2' : 'col-span-2'}>
        <ImagePlayer />
      </div>
      <div class="sm:col-span-2">
        <Button on:click={toggleLcmLive} {disabled} classList={'text-lg my-1 p-2'}>
          {#if isLCMRunning}
            Stop
          {:else}
            Start
          {/if}
        </Button>
        <PipelineOptions {pipelineParams} {pipelineInfo}></PipelineOptions>
      </div>
    </article>
  {:else}
    <!-- loading -->
    <div class="flex items-center justify-center gap-3 py-48 text-2xl">
      <Spinner classList={'animate-spin opacity-50'}></Spinner>
      <p>Loading...</p>
    </div>
  {/if}
</main>

<style lang="postcss">
  :global(html) {
    @apply text-black dark:bg-gray-900 dark:text-white;
  }
</style>
