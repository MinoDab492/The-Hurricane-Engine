import unreal

global current_job_index

# Python needs to keep a reference to the delegate, as the delegate itself
# only has a weak reference to the Python function, so this lets the Python
# reference collector see the callback function and keep it alive.
global delegate_callback

def GeneratePreStreamingForCurrentQueue():
    global current_job_index  
    current_job_index = 0
    
    queue = unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem).get_queue()
    if len(queue.get_jobs()) == 0:
        unreal.log_warning("No jobs currently added to the Movie Render Queue to generate assets for!")
        return
        
    prestreaming_subsystem = unreal.get_editor_subsystem(unreal.CinePrestreamingEditorSubsystem)
    
    global delegate_callback
    delegate_callback = prestreaming_subsystem.on_asset_generated
    delegate_callback.add_callable_unique(OnIndividualJobFinished)
    
    BuildPreStreamingForJobByIndex(0)
    
def BuildPreStreamingForJobByIndex(in_job_index):
    unreal.log_warning("BuildPreStreamingForJobByIndex")
    queue = unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem).get_queue()
    if in_job_index < 0 or in_job_index >= len(queue.get_jobs()):
        unreal.log_error("Out of bound job index!")
        return
    
    global current_job_index
    current_job_index = in_job_index
    
    job = queue.get_jobs()[current_job_index]
    args = unreal.CinePrestreamingGenerateAssetArgs()
    args.resolution = unreal.IntPoint(3840, 2160)
    args.sequence = job.sequence
    args.map = job.map
    
    prestreaming_subsystem = unreal.get_editor_subsystem(unreal.CinePrestreamingEditorSubsystem)
    prestreaming_subsystem.generate_prestreaming_asset(args)
    
def OnIndividualJobFinished(generation_args):
    unreal.log("Assets generated for job.")
    
    global current_job_index
    queue = unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem).get_queue()
    
    if current_job_index + 1 < len(queue.get_jobs()):
        BuildPreStreamingForJobByIndex(current_job_index + 1)
    else:
        unreal.log("Finished generating assets for all jobs in the queue.")

# Actually run the job
GeneratePreStreamingForCurrentQueue()