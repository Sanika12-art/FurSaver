document.addEventListener('DOMContentLoaded', function() {
    // Cache DOM elements
    const analysisForm = document.getElementById('analysisForm');
    const symptomsContainer = document.getElementById('symptomsContainer');
    const loadingOverlay = document.querySelector('.loading-overlay');
    const petTypeRadios = document.querySelectorAll('input[name="petType"]');

    // Initialize loading overlay
    if (loadingOverlay) {
        loadingOverlay.style.display = 'none';
    }
    
    // Symptom options for Cats and Dogs
// Replace the symptoms object in diseasepredict.js:
const symptoms = {
    Cat: [
        // Symptoms for Cat Diseases
        { id: 'redness', label: 'Redness', icon: 'fa-circle' },
        { id: 'swelling', label: 'Swelling', icon: 'fa-circle-plus' },
        { id: 'itching', label: 'Itching', icon: 'fa-hand-dots' },
        { id: 'inflammation', label: 'Inflammation', icon: 'fa-circle-plus' },
        { id: 'hairLoss', label: 'Hair Loss', icon: 'fa-paintbrush' },
        { id: 'circularHairLoss', label: 'Circular Hair Loss', icon: 'fa-circle-dot' },
        { id: 'scalyPatches', label: 'Scaly Patches', icon: 'fa-layer-group' },
        { id: 'crustySkin', label: 'Crusty Skin', icon: 'fa-virus-covid' },
        { id: 'patchyHairLoss', label: 'Patchy Hair Loss', icon: 'fa-droplet' },
        { id: 'intenseItching', label: 'Intense Itching', icon: 'fa-hand-dots' },
        { id: 'thickenedSkin', label: 'Thickened Skin', icon: 'fa-layer-group' }
    ],
    Dog: [
        // Symptoms for Dog Diseases
        { id: 'cloudyEyes', label: 'Cloudy Eyes', icon: 'fa-eye' },
        { id: 'visionLoss', label: 'Vision Loss', icon: 'fa-eye-slash' },
        { id: 'eyeDischarge', label: 'Eye Discharge', icon: 'fa-droplet' },
        { id: 'redEyes', label: 'Red Eyes', icon: 'fa-eye' },
        { id: 'discharge', label: 'Discharge', icon: 'fa-droplet' },
        { id: 'swelling', label: 'Swelling', icon: 'fa-circle-plus' },
        { id: 'pus', label: 'Pus', icon: 'fa-virus-covid' },
        { id: 'wound', label: 'Wound', icon: 'fa-bandage' },
        { id: 'nasalDischarge', label: 'Nasal Discharge', icon: 'fa-droplet' },
        { id: 'crustyNose', label: 'Crusty Nose', icon: 'fa-virus-covid' },
        { id: 'noseIrritation', label: 'Nose Irritation', icon: 'fa-circle-plus' },
        { id: 'eyeIrritation', label: 'Eye Irritation', icon: 'fa-circle-plus' },
        { id: 'intenseItching', label: 'Intense Itching', icon: 'fa-hand-dots' },
        { id: 'scabs', label: 'Scabs', icon: 'fa-virus-covid' },
        { id: 'constantScratching', label: 'Constant Scratching', icon: 'fa-hand-back-fist' },
        { id: 'inflammation', label: 'Inflammation', icon: 'fa-circle-plus' },
        { id: 'redness', label: 'Redness', icon: 'fa-circle' },
        { id: 'scaling', label: 'Scaling', icon: 'fa-layer-group' },
        { id: 'crustySkin', label: 'Crusty Skin', icon: 'fa-virus-covid' },
        { id: 'circularLesions', label: 'Circular Lesions', icon: 'fa-circle-dot' },
        { id: 'hairLoss', label: 'Hair Loss', icon: 'fa-paintbrush' }
    ]
};

const cat_symptom_mapping = {
    "dermatitis": ["Redness", "Swelling", "Itching", "Inflammation", "Hair Loss"],
    "ringworm": ["Circular Hair Loss", "Scaly Patches", "Redness", "Crusty Skin"],
    "scabies": ["Intense Itching", "Crusty Skin", "Hair Loss", "Redness"],
    "Mange": ["Patchy Hair Loss", "Thickened Skin", "Intense Itching"],
    "Healthy skin": ["No Symptoms"]
};

const dog_symptom_mapping = {
    "Cataratas": ["Cloudy Eyes", "Vision Loss", "Eye Discharge"],
    "Conjuntivitis": ["Red Eyes", "Discharge", "Swelling", "Eye Irritation"],
    "Infección Bacteriana": ["Pus", "Swelling", "Wound", "Redness"],
    "PyodermaNasal": ["Nasal Discharge", "Crusty Nose", "Nose Irritation"],
    "Sarna": ["Intense Itching", "Scabs", "Hair Loss", "Redness"],
    "dermatitis": ["Inflammation", "Redness", "Scaling", "Itching"],
    "flea_allergy": ["Constant Scratching", "Hair Loss", "Scabs"],
    "ringworm": ["Circular Lesions", "Hair Loss", "Scaling"],
    "scabies": ["Crusty Skin", "Intense Itching", "Hair Loss"],
    "Healthy skin": ["No Symptoms"]
};
    // Handle image previews
    function setupImagePreview(inputId, previewId) {
        const input = document.getElementById(inputId);
        const preview = document.getElementById(previewId);
        const container = input.closest('.upload-card');

        if (!input || !preview) return;

        input.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(event) {
                    preview.src = event.target.result;
                    preview.style.display = 'block';
                    preview.closest('.image-preview-wrapper').style.display = 'block';
                    container.classList.add('has-preview');
                };
                reader.readAsDataURL(file);
            }
        });

        // Handle drag and drop
        const dropZone = input.closest('.file-upload-container');
        
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, preventDefaults, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, highlight, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, unhighlight, false);
        });

        function highlight(e) {
            dropZone.classList.add('drag-over');
        }

        function unhighlight(e) {
            dropZone.classList.remove('drag-over');
        }

        dropZone.addEventListener('drop', handleDrop, false);

        function handleDrop(e) {
            const dt = e.dataTransfer;
            const file = dt.files[0];
            input.files = dt.files;
            
            if (file) {
                const reader = new FileReader();
                reader.onload = function(event) {
                    preview.src = event.target.result;
                    preview.style.display = 'block';
                    preview.closest('.image-preview-wrapper').style.display = 'block';
                    container.classList.add('has-preview');
                };
                reader.readAsDataURL(file);
            }
        }
    }

    // Initialize image previews
    setupImagePreview('fullAnimalImage', 'fullAnimalPreview');
    setupImagePreview('diseaseImage', 'diseasePreview');

    // Initialize remove image buttons
    document.querySelectorAll('.remove-image').forEach(button => {
        button.addEventListener('click', function() {
            const targetId = this.dataset.target;
            const input = document.getElementById(targetId);
            const previewWrapper = this.closest('.image-preview-wrapper');
            const container = input.closest('.file-upload-container');
            
            input.value = '';
            previewWrapper.style.display = 'none';
            container.classList.remove('has-preview');
        });
    });

    // Update symptoms based on pet type
    function updateSymptoms(petType) {
        symptomsContainer.innerHTML = '';
        if (symptoms[petType]) {
            symptoms[petType].forEach(symptom => {
                const symptomItem = document.createElement('div');
                symptomItem.className = 'symptom-item';
                
                symptomItem.innerHTML = `
                    <input type="checkbox" id="${symptom.id}" name="symptoms" value="${symptom.label}">
                    <label for="${symptom.id}">
                        <i class="fas ${symptom.icon} mr-2"></i>
                        ${symptom.label}
                    </label>
                `;
                
                symptomsContainer.appendChild(symptomItem);

                // Add animation
                setTimeout(() => {
                    symptomItem.style.opacity = '1';
                    symptomItem.style.transform = 'translateY(0)';
                }, 50);
            });
        }
    }

    // Pet type change handler
    petTypeRadios.forEach(radio => {
        radio.addEventListener('change', () => {
            updateSymptoms(radio.value);
        });
    });

    // Form submission handler
    analysisForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Validate form
        const requiredFields = analysisForm.querySelectorAll('[required]');
        let isValid = true;
        
        requiredFields.forEach(field => {
            if (!field.value) {
                isValid = false;
                field.classList.add('is-invalid');
            } else {
                field.classList.remove('is-invalid');
            }
        });

        // Check if pet type is selected
        const petType = document.querySelector('input[name="petType"]:checked');
        if (!petType) {
            isValid = false;
            document.querySelector('.pet-type-buttons').classList.add('is-invalid');
        }

        // Check if at least one symptom is selected
        const selectedSymptoms = document.querySelectorAll('input[name="symptoms"]:checked');
        if (selectedSymptoms.length === 0) {
            isValid = false;
            const symptomsError = document.createElement('div');
            symptomsError.className = 'alert alert-danger mt-3';
            symptomsError.innerHTML = 'Please select at least one symptom';
            symptomsContainer.parentNode.insertBefore(symptomsError, symptomsContainer.nextSibling);
            
            setTimeout(() => {
                symptomsError.remove();
            }, 3000);
        }

        if (isValid) {
            // Show loading overlay
            if (loadingOverlay) {
                loadingOverlay.style.display = 'flex';
            }
            
            // Submit form
            this.submit();
        }
    });

    // Reset form handler
    analysisForm.addEventListener('reset', function() {
        // Clear image previews
        document.querySelectorAll('.image-preview-wrapper').forEach(wrapper => {
            wrapper.style.display = 'none';
        });

        // Clear upload success states
        document.querySelectorAll('.file-upload-container').forEach(container => {
            container.classList.remove('has-preview');
        });

        // Clear symptoms
        symptomsContainer.innerHTML = '';

        // Remove any error states
        document.querySelectorAll('.is-invalid').forEach(field => {
            field.classList.remove('is-invalid');
        });

        // Reset file inputs
        document.querySelectorAll('.file-input').forEach(input => {
            input.value = '';
        });
    });
});