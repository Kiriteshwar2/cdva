# 🔍 CIF Data Flow Debugging Guide

## Overview
This document traces the complete CIF data pipeline from backend generation through to 3D rendering.

---

## 📋 Data Flow Stages

### STAGE 1: Backend Generation
**File:** `backend/app/api/routes/generation.py`  
**Function:** `generate_crystal()`

**What happens:**
- Backend generates crystal structure
- Creates CIF string via `CifWriter()`
- Stores in `candidate.cif_string`

**Debug Output (Backend Console):**
```
==================================================================
[BACKEND /generate] CIF Generation Debug
==================================================================
CIF String Type: <class 'str'>
CIF String Length: 450 characters
CIF Valid (starts with 'data_'): True
CIF First 150 chars:
data_test
_cell_length_a 5.0
...
==================================================================
```

**✅ Success Criteria:**
- [ ] CIF String Type shows `<class 'str'>`
- [ ] CIF String Length > 0
- [ ] CIF Valid shows `True`
- [ ] CIF starts with "data_"

**❌ If Failed:**
- Check inference.py `_build_candidate()` returns valid CIF
- Verify `CifWriter` is called correctly
- Check for RuntimeError in generation

---

### STAGE 2: API Response
**File:** `backend/app/api/routes/generation.py`  
**Database:** MongoDB stores generation document

**Response Structure:**
```json
{
  "id": "...",
  "output_cif": "data_...",  ← CRITICAL FIELD
  "structure": {...},
  "metadata": {...},
  "created_at": "..."
}
```

**✅ Success Criteria:**
- [ ] Response includes `output_cif` field
- [ ] `output_cif` is NOT null
- [ ] `output_cif` is a string (not object)
- [ ] `output_cif` starts with "data_"

---

### STAGE 3: Frontend API Call
**File:** `frontend/src/services/api.js`  
**Function:** `createGeneration()`

**Debug Output (Browser Console):**
```
[FRONTEND API] Generation Response Received
Response structure: ['id', 'output_cif', 'structure', 'metadata', ...]
output_cif exists: true
output_cif type: string
output_cif length: 450
CIF preview: data_...
CIF valid format: true
======================================================================
```

**✅ Success Criteria:**
- [ ] "Generation Response Received" message appears
- [ ] `output_cif exists: true`
- [ ] `output_cif type: string`
- [ ] `output_cif length` > 0
- [ ] `CIF valid format: true`

**❌ If Failed:**
- Backend not returning CIF in response
- Check `/generate` endpoint response format
- Verify MongoDB storage includes `output_cif`

---

### STAGE 4: StructureViewer Component
**File:** `frontend/src/components/StructureViewer.jsx`

**Debug Output (Browser Console):**
```
[FRONTEND StructureViewer] CIF Prop Check
Generation object keys: ['id', 'output_cif', 'structure', 'metadata', ...]
output_cif field found: true
cifString length: 450
cifString valid format: true
cifString preview: data_...
======================================================================
```

**✅ Success Criteria:**
- [ ] "CIF Prop Check" message appears
- [ ] `output_cif field found: true`
- [ ] `cifString length` > 0
- [ ] `cifString valid format: true`

**❌ If Failed:**
- Check destructuring: `const { output_cif } = generation`
- Verify property name is exactly `output_cif` (not `cif` or `cif_string`)
- Check response from Stage 3

---

### STAGE 5: 3D Viewer Component
**File:** `frontend/src/components/StructureViewer3D.jsx`

**Debug Output (Browser Console):**
```
[3D VIEWER] Props Received
cifString type: string
cifString length: 450
Using TEST_CIF: false
isValidCif result: true
✓ CIF ready, first 200 chars:
data_test
_cell_length_a 5.0
...
======================================================================
```

**✅ Success Criteria:**
- [ ] "[3D VIEWER] Props Received" message appears
- [ ] `cifString type: string`
- [ ] `cifString length` > 0
- [ ] `isValidCif result: true`
- [ ] CIF content is displayed

**❌ If Failed:**
- Check StructureViewer passes `cifString={cifString}` prop
- Verify CIF is not empty string
- Check if fallback "No crystal structure available" message shows

---

## 🧪 Testing Scenarios

### Scenario 1: Test Backend → API Flow
**Action:**
1. Start backend server
2. Generate a crystal via API (e.g., using curl or Postman)
3. Check backend console for Stage 1 debug output
4. Verify response includes `output_cif`

**Example curl:**
```bash
curl -X POST http://localhost:8000/generate \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "elements": ["C", "O"],
    "num_atoms": 8,
    "candidate_pool_size": 16,
    "max_attempts": 32
  }'
```

### Scenario 2: Test Frontend Data Reception
**Action:**
1. Start frontend dev server
2. Trigger generation via UI
3. Check browser console for:
   - Stage 3: "Generation Response Received"
   - Stage 4: "CIF Prop Check"
   - Stage 5: "Props Received"

### Scenario 3: Test Hardcoded CIF (Viewer Only)
**Action:**
1. Edit `StructureViewer3D.jsx`
2. Uncomment: `const activeCif = TEST_CIF;`
3. Comment: `const activeCif = cifString;`
4. Reload frontend
5. Should render a simple 2-atom structure

**Expected:**
- Spinner shows "Rendering structure..."
- 3D viewer appears with ball-and-stick model
- Can rotate/zoom/pan
- View mode selector works

---

## 🛠️ Debugging Checklist

### When 3D Viewer is Blank:

- [ ] **Backend Console:**
  - Check for "[BACKEND /generate]" debug output
  - Verify CIF starts with "data_"
  - Check CIF length > 0

- [ ] **Browser Console:**
  - Check for "[FRONTEND API]" debug output
  - Search for "ERROR" or "WARNING" messages
  - Check for "[FRONTEND StructureViewer]" debug output
  - Check for "[3D VIEWER]" debug output

- [ ] **Network Tab:**
  - Verify `/generate` request succeeds (200 status)
  - Expand response body
  - Find `output_cif` field
  - Verify it contains valid CIF data

- [ ] **React DevTools:**
  - Inspect `<StructureViewer />` component
  - Check props: does it receive `generation.output_cif`?
  - Inspect `<StructureViewer3D />` component
  - Check props: does `cifString` have content?

---

## 🚨 Common Issues & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| "No crystal structure available" | `cifString` is empty | Check Stage 4 debug output |
| 3D viewer loads but blank | CIF not rendering in 3Dmol.js | Test with hardcoded CIF |
| "Failed to render: ..." error | Invalid CIF format | Check Stage 5: `isValidCif` result |
| API returns 500 error | Backend generation failed | Check backend console for errors |
| API returns `output_cif: null` | CIF generation bug | Check Stage 1 debug output |
| Stats show but viewer blank | CIF not passed to component | Check `output_cif` field name |

---

## 📊 Expected Output Timeline

1. **Generate button clicked** → Shows loading spinner

2. **Backend processes** (1-30 seconds):
   - Backend console shows Stage 1 debug
   - Generation completes

3. **Frontend receives response** (immediate):
   - Browser console shows Stage 3 debug
   - Browser console shows Stage 4 debug

4. **Viewer initializes** (1-2 seconds):
   - Browser console shows Stage 5 debug
   - Shows "Rendering structure..." overlay
   - 3Dmol.js library loads from CDN

5. **Structure renders** (1-5 seconds):
   - Overlay disappears
   - 3D structure appears in viewer
   - Can interact with controls

---

## 🔗 Key Files to Monitor

```
Backend:
  ├── backend/app/api/routes/generation.py        [Add CIF debug]
  ├── backend/app/services/inference.py           [Generates CIF]
  └── backend/app/schemas/generation.py           [Response structure]

Frontend:
  ├── frontend/src/services/api.js                [API call debug]
  ├── frontend/src/components/StructureViewer.jsx [Prop passing debug]
  └── frontend/src/components/StructureViewer3D.jsx [Viewer debug]

Database:
  └── MongoDB: generations collection              [Check output_cif field]
```

---

## ✅ Production Checklist

Once viewer is rendering correctly:

- [ ] Remove or comment out all debug console.log statements
- [ ] Test with multiple crystal types
- [ ] Test error cases (invalid CIF, network failure)
- [ ] Verify performance (viewer loads in <5 seconds)
- [ ] Check mobile responsiveness
- [ ] Verify screenshot download works
- [ ] Test all view modes (stick, ball_stick, space_filling)
