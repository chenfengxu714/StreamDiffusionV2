
import { derived, writable, get, type Writable, type Readable } from 'svelte/store';

export const pipelineValues: Writable<Record<string, any>> = writable({});
export const promptValues: Writable<Record<string, any>> = writable({});
export const deboucedPipelineValues: Readable<Record<string, any>>
    = derived(pipelineValues, ($pipelineValues, set) => {
        const debounced = setTimeout(() => {
            set($pipelineValues);
        }, 100);
        return () => clearTimeout(debounced);
    });

export const getPipelineValues = () => get(pipelineValues);
export const getPromptValues = () => get(promptValues);

// Function to send prompt values to the pipeline
export const sendPromptValues = () => {
    const currentPromptValues = get(promptValues);
    pipelineValues.set({ ...get(pipelineValues), ...currentPromptValues });
};