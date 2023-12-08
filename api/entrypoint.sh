if [-z "${ENERGYAI_API_PORT}"]
  then
    uvicorn 'api:app' --port 50001 --host=0.0.0.0 --workers 4
  else
    uvicorn 'api:app' --port "${ENERGYAI_API_PORT}" --host=0.0.0.0 --workers 4
fi
